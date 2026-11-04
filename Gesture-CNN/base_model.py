import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

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
    
def get_pytorch_loaders(batch_size=16):
    full_dataset = ASLDataset()
    total_samples = len(full_dataset)
    
    # 75% Train / 25% Test Split
    train_size = int(0.75 * total_samples)
    test_size = total_samples - train_size
    
    # Deterministic split via fixed generator seed
    train_data, test_data = random_split(
        full_dataset, 
        [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_size, test_size

if __name__ == "__main__":
    train_loader, test_loader, n_train, n_test = get_pytorch_loaders()
    print(f"PyTorch Dataset Split (75/25):")
    print(f"  Training samples: {n_train} | Batches: {len(train_loader)}")
    print(f"  Testing samples:  {n_test}  | Batches: {len(test_loader)}")

# Quick verification test
#if __name__ == "__main__":
#    dataset = ASLDataset()
#    loader = DataLoader(dataset, batch_size=16, shuffle=True)
#    batch_x, batch_y = next(iter(loader))
#    print(f"Loaded {len(dataset)} total samples.")
#    print(f"PyTorch Batch Shape: {batch_x.shape} (Batch, Channels, H, W)")