import os

# 1. Disable oneDNN floating-point notices
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Filter out TensorFlow C++ info logs (0 = all, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = filter all)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

TARGET_CLASSES = ["A", "B", "I", "O", "U", "V"]
SAMPLES_PER_CLASS = 50
IMG_SIZE = (48, 48)

def load_and_split_asl_dataset(test_r = 0.25, batch_size=16):
    images, labels = [], []
    for class_idx, letter in enumerate(TARGET_CLASSES):
        folder = os.path.join("dataset_images", f"{letter}_samples")
        files = sorted(glob.glob(os.path.join(folder, "*.*")))[:SAMPLES_PER_CLASS]
        for f in files:
            with Image.open(f) as img:
                img = img.convert('L').resize(IMG_SIZE, Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(class_idx)

    # Convert to NumPy arrays; add channel dimension: (N, 48, 48, 1)
    X = np.expand_dims(np.array(images, dtype=np.float32), axis=-1)
    y = np.array(labels, dtype=np.int32)
    
    # Test and Train split, stratified to ensure near equal inclusion of classes in training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_r, stratify=y)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    return (X_train, y_train, train_dataset), (X_test, y_test, test_dataset)

# CNN Architecture, works for 48x48 image, using 3x3 filters
def build_tinyml_cnn(input_shape=(48, 48, 1), num_classes=6):
    model = tf.keras.Sequential([
        # Explicit input layer to clear Keras 3 warnings
        tf.keras.layers.Input(shape=input_shape),
        
        # Layer 1: Stride-2 immediately reduces 48x48 to 24x24, cutting activation SRAM footprint
        tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same'),
        tf.keras.layers.ReLU(max_value=6.0),  # ReLU6 prevents INT8 overflow
        
        # Layer 2: Feature extraction
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same'),
        tf.keras.layers.ReLU(max_value=6.0),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # Downsamples 24x24 -> 12x12
        
        # Layer 3: Consolidates spatial representation
        tf.keras.layers.Conv2D(32, (3, 3), strides=1, padding='same'),
        tf.keras.layers.ReLU(max_value=6.0),
        
        # Collapse spatial grid to a compact 2x2 quadrant layout without parameter explosion
        tf.keras.layers.AveragePooling2D(pool_size=(6, 6)),  # Downsamples 12x12 -> 2x2
        tf.keras.layers.Flatten(),                           # 2 * 2 * 32 = 128 inputs to dense layer
        
        # Classification Head: 6 classes (A, B, I, O, U, V)
        tf.keras.layers.Dense(num_classes)
    ])
    return model

# Diagnostic plotting helpers
def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r--', label='Validation Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(epochs, acc, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=200)
    print("Saved training curves to 'training_history.png'.")
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=TARGET_CLASSES, 
           yticklabels=TARGET_CLASSES,
           title='Test Split Confusion Matrix',
           ylabel='True Gesture',
           xlabel='Predicted Gesture')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200)
    print("Saved confusion matrix to 'confusion_matrix.png'.")
    plt.close()

# INT8 post-training quantization conversion helper
def quantize_model_to_tflite(model, X_train, output_path="asl_tinyml_model_quant_int8.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Generator providing representative real image samples to calibrate INT8 scale and zero points
    def representative_dataset():
        for i in range(min(100, len(X_train))):
            yield [np.expand_dims(X_train[i], axis=0)]

    converter.representative_dataset = representative_dataset
    # Enforce pure full-integer quantization (inputs, weights, activations, outputs)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)

    print(f"Saved INT8 quantized FlatBuffer to '{output_path}'.")
    print(f"Quantized Model Size: {len(tflite_quant_model) / 1024:.2f} KB")
    return output_path

# C-Header byte dumper compatible with Arduino Nano 33 BLE (TensorFlow Lite for Microcontrollers)
def hex_to_c_header(tflite_path, header_path="model_data.h", array_name="model_data"):
    with open(tflite_path, "rb") as f:
        bytes_data = f.read()

    header_content = []
    header_content.append(f"// Auto-generated Arduino C Header for TinyML ASL Gesture Recognition\n")
    header_content.append(f"#ifndef {array_name.upper()}_H\n")
    header_content.append(f"#define {array_name.upper()}_H\n\n")
    header_content.append(f"// 4-byte memory alignment attribute required by TFLite MicroInterpreter\n")
    header_content.append(f"const unsigned char {array_name}[] __attribute__((aligned(4))) = {{\n  ")

    hex_lines = []
    for i, byte in enumerate(bytes_data):
        hex_lines.append(f"0x{byte:02x}")
        if (i + 1) % 12 == 0 and (i + 1) < len(bytes_data):
            hex_lines.append(",\n  ")
        elif (i + 1) < len(bytes_data):
            hex_lines.append(", ")

    header_content.append("".join(hex_lines))
    header_content.append(f"\n}};\n\n")
    header_content.append(f"const int {array_name}_len = {len(bytes_data)};\n\n")
    header_content.append(f"#endif // {array_name.upper()}_H\n")

    with open(header_path, "w") as f:
        f.writelines(header_content)

    print(f"Saved C array header to '{header_path}'. Length: {len(bytes_data)} bytes.")

# Main training, evaluation, and C export execution
if __name__ == "__main__":
    (X_tr, y_tr, train_ds), (X_te, y_te, test_ds) = load_and_split_asl_dataset()
    print("=" * 60)
    print(f"Dataset Split (75/25):")
    print(f"  Training Set: {X_tr.shape[0]} images | Tensor Shape: {X_tr.shape}")
    print(f"  Testing Set:   {X_te.shape[0]} images | Tensor Shape: {X_te.shape}")
    print("=" * 60)
    
    # 1. Build and Compile Model
    model = build_tinyml_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # 2. Train Model
    print("\nStarting Training (40 Epochs, Batch Size 16)...")
    history = model.fit(
        X_tr, y_tr,
        epochs=40,
        batch_size=16,
        validation_data=(X_te, y_te),
        verbose=1
    )
    
    # 3. Evaluate Final Test Performance
    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    print("\n" + "=" * 60)
    print(f"FINAL HELD-OUT TEST RESULTS:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 60)

    # 4. Save Model Weights (Keras Checkpoint)
    model.save("asl_tinyml_model.keras")
    print("Saved trained weights to 'asl_tinyml_model.keras'.")

    # 5. Generate Diagnostic Plots
    plot_training_history(history)
    
    y_pred_logits = model.predict(X_te)
    y_pred = np.argmax(y_pred_logits, axis=1)
    plot_confusion_matrix(y_te, y_pred)

    # 6. Quantize to INT8 FlatBuffer and Export Arduino C Header
    print("\nStarting INT8 Post-Training Quantization...")
    tflite_file = quantize_model_to_tflite(model, X_tr)
    hex_to_c_header(tflite_file, header_path="model_data.h")

# Main training and evaluation execution
#if __name__ == "__main__":
#    (X_tr, y_tr, train_ds), (X_te, y_te, test_ds) = load_and_split_asl_dataset()
#    print("=" * 60)
#    print(f"Dataset Split (75/25):")
#    print(f"  Training Set: {X_tr.shape[0]} images | Tensor Shape: {X_tr.shape}")
#    print(f"  Testing Set:   {X_te.shape[0]} images | Tensor Shape: {X_te.shape}")
#    print("=" * 60)
#    
#    # 1. Build and Compile Model
#    model = build_tinyml_cnn()
#    model.compile(
#        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#        metrics=['accuracy']
#    )
#    
#    # 2. Train Model
#    print("\nStarting Training (40 Epochs, Batch Size 16)...")
#    history = model.fit(
#        X_tr, y_tr,
#        epochs=40,
#        batch_size=16,
#        validation_data=(X_te, y_te),
#        verbose=1
#    )
#    
#    # 3. Evaluate Final Test Performance
#    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
#    print("\n" + "=" * 60)
#    print(f"FINAL HELD-OUT TEST RESULTS:")
#    print(f"  Test Loss:     {test_loss:.4f}")
#    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
#    print("=" * 60)
#
#    # 4. Save Model Weights
#    os.makedirs("models", exist_ok=True)
#    model.save("asl_tinyml_model.keras")
#    print("Saved trained weights to 'asl_tinyml_model.keras'.")
#
#    # 5. Generate Diagnostic Plots
#    plot_training_history(history)
#    
#    y_pred_logits = model.predict(X_te)
#    y_pred = np.argmax(y_pred_logits, axis=1)
#    plot_confusion_matrix(y_te, y_pred)

# Execution and memory footprint diagnostics
#if __name__ == "__main__":
#    (X_tr, y_tr, train_ds), (X_te, y_te, test_ds) = load_and_split_asl_dataset()
#    print("=" * 60)
#    print(f"Dataset Split (75/25):")
#    print(f"  Training Set: {X_tr.shape[0]} images | Tensor Shape: {X_tr.shape}")
#    print(f"  Testing Set:   {X_te.shape[0]} images | Tensor Shape: {X_te.shape}")
#    print("=" * 60)
#    
#    # Instantiate and inspect model
#    model = build_tinyml_cnn()
#    print("\nTinyML Model Architecture & Parameter Summary:")
#    model.summary()

# Size Testing for image arrays for model training and testing
#if __name__ == "__main__":
#    (X_tr, y_tr, train_ds), (X_te, y_te, test_ds) = load_and_split_asl_dataset()
#    print(f"TensorFlow Dataset Split (75/25):")
#    print(f"  X_train shape: {X_tr.shape} | y_train shape: {y_tr.shape}")
#    print(f"  X_test shape:  {X_te.shape}  | y_test shape:  {y_te.shape}")

# Quick verification test
#if __name__ == "__main__":
#    X, y, dataset = load_and_split_asl_dataset()
#    for batch_x, batch_y in dataset.take(1):
#        print(f"Loaded {len(y)} total samples.")
#        print(f"TensorFlow Batch Shape: {batch_x.shape} (Batch, H, W, Channels)")