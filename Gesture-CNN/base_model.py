import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TARGET_CLASSES = ["A", "B", "I", "O", "U", "V"]
SAMPLES_PER_CLASS = 50
IMG_SIZE = (48, 48)

class ASLDataset(Dataset):
    def __init__(self, root_dir="dataset_images"):
        self.samples = []
        for class_idx, letter in enumerate(TARGET_CLASSES):
            folder = os.path.join(root_dir, f"{letter}_samples")
            files = sorted(glob.glob(os.path.join(folder, "*.*")))[:SAMPLES_PER_CLASS]
            for f in files:
                self.samples.append((f, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('L').resize(IMG_SIZE, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            x = torch.tensor(arr).unsqueeze(0)  # Shape: (1, 48, 48)
            y = torch.tensor(label, dtype=torch.long)
        return x, y

# Quick verification test
if __name__ == "__main__":
    dataset = ASLDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Loaded {len(dataset)} total samples.")
    print(f"PyTorch Batch Shape: {batch_x.shape} (Batch, Channels, H, W)")