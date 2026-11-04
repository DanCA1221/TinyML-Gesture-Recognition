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
    
    # Test and Train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_r, stratify=y)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    return (X_train, y_train, train_dataset), (X_test, y_test, test_dataset)

if __name__ == "__main__":
    (X_tr, y_tr, train_ds), (X_te, y_te, test_ds) = load_and_split_asl_dataset()
    print(f"TensorFlow Dataset Split (75/25):")
    print(f"  X_train shape: {X_tr.shape} | y_train shape: {y_tr.shape}")
    print(f"  X_test shape:  {X_te.shape}  | y_test shape:  {y_te.shape}")

# Quick verification test
#if __name__ == "__main__":
#    X, y, dataset = load_and_split_asl_dataset()
#    for batch_x, batch_y in dataset.take(1):
#        print(f"Loaded {len(y)} total samples.")
#        print(f"TensorFlow Batch Shape: {batch_x.shape} (Batch, H, W, Channels)")