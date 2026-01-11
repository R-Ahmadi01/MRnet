import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------- SETTINGS ---------
ROOT = Path(r"C:\Users\ihadl\Documents\MRNet-v1.0\MRNet-v1.0")
TASK = "abnormal"     # "abnormal" or "acl" or "meniscus"
PLANE = "axial"       # "axial" or "coronal" or "sagittal"
NUM_SLICES = 16       # fixed number of slices (deterministic)
BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-4
# ---------------------------


def load_labels(csv_path: Path):
    # CSV format: exam_id,label
    labels = {}
    with csv_path.open("r") as f:
        header = f.readline()  # skip header
        for line in f:
            exam_id, lab = line.strip().split(",")
            labels[exam_id] = int(lab)
    return labels


class MRNetDataset(Dataset):
    def __init__(self, root: Path, split: str, plane: str, task: str, num_slices: int):
        self.root = root
        self.split = split
        self.plane = plane
        self.task = task
        self.num_slices = num_slices

        self.npy_paths = sorted((root / split / plane).glob("*.npy"))
        if len(self.npy_paths) == 0:
            raise FileNotFoundError(
                f"No .npy files found in: {root / split / plane}")

        self.labels_path = root / f"{split}-{task}.csv"
        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {self.labels_path}")

        self.labels = load_labels(self.labels_path)

        # keep only exams that have labels
        self.npy_paths = [p for p in self.npy_paths if p.stem in self.labels]

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        p = self.npy_paths[idx]
        vol = np.load(p, mmap_mode="r")          # shape usually (S, H, W)
        S = vol.shape[0]

        # Deterministic slice selection (no randomness)
        idxs = np.linspace(0, S - 1, self.num_slices).round().astype(int)
        x = vol[idxs].astype(np.float32)         # NO normalization
        x = torch.from_numpy(x).unsqueeze(1)     # (num_slices, 1, H, W)

        y = torch.tensor(self.labels[p.stem], dtype=torch.float32)
        return x, y


class SliceCNN(nn.Module):
    """
    Takes input (B, S, 1, H, W)
    Runs a small 2D CNN on each slice, then max-pools over slices -> 1 logit per exam.
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # (B,64,1,1)
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.cnn(x).view(B * S, 64)
        feats = feats.view(B, S, 64)
        pooled, _ = feats.max(dim=1)            # max over slices
        logit = self.fc(pooled).squeeze(1)      # (B,)
        return logit


def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == y).float().mean().item()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)   # (B,S,1,H,W)
        y = y.to(device)   # (B,)

        logits = model(x)
        loss = loss_fn(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)  # helps stability
            optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_from_logits(logits.detach(), y) * bs
        n += bs

    return total_loss / n, total_acc / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = MRNetDataset(ROOT, "train", PLANE, TASK, NUM_SLICES)
    valid_ds = MRNetDataset(ROOT, "valid", PLANE, TASK, NUM_SLICES)

    # num_workers=0 is safest on Windows
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Train samples:", len(train_ds))
    print("Valid samples:", len(valid_ds))

    model = SliceCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(
            model, valid_loader, optimizer, device, train=False)

        print(f"Epoch {epoch}: "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"valid loss {va_loss:.4f} acc {va_acc:.4f}")


if __name__ == "__main__":
    main()
