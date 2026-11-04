import os

# 1. Disable oneDNN floating-point notices
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Filter out TensorFlow C++ info logs (0 = all, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = filter all)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
from PIL import Image
import numpy as np
import tensorflow as tf

TARGET_CLASSES = ["A", "B", "I", "O", "U", "V"]
SAMPLES_PER_CLASS = 50
IMG_SIZE = (48, 48)

def load_asl_dataset(root_dir="dataset_images"):
    images, labels = [], []
    for class_idx, letter in enumerate(TARGET_CLASSES):
        folder = os.path.join(root_dir, f"{letter}_samples")
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
    
    # Wrap in standard tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(y)).batch(16)
    return X, y, dataset

# Quick verification test
if __name__ == "__main__":
    X, y, dataset = load_asl_dataset()
    for batch_x, batch_y in dataset.take(1):
        print(f"Loaded {len(y)} total samples.")
        print(f"TensorFlow Batch Shape: {batch_x.shape} (Batch, H, W, Channels)")