"""
Training Script for the Star CNN

Loads the synthetic dataset, trains ResNet-18 on attitude regression,
and saves the best model checkpoint.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from node_b_perception.star_cnn import StarCNN


# ──────────────────────────────────────────────
# Custom Dataset
# ──────────────────────────────────────────────
class StarImageDataset(Dataset):
    """Loads .npy star images + Euler angle labels."""

    def __init__(self, images_dir, labels, split_mask):
        self.images_dir = images_dir
        self.indices = np.where(split_mask)[0]
        self.labels = labels[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = np.load(
            os.path.join(self.images_dir, f"{real_idx:06d}.npy")
        )
        # Add channel dimension: (224, 224) → (1, 224, 224)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        label_tensor = torch.from_numpy(self.labels[idx]).float()
        return img_tensor, label_tensor


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────
def train(data_dir=None, epochs=None, batch_size=None,
          lr=None, save_path=None):

    if data_dir is None:
        data_dir = config.TRAINING_IMAGES_DIR
    if epochs is None:
        epochs = config.NUM_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if lr is None:
        lr = config.LEARNING_RATE
    if save_path is None:
        save_path = config.MODEL_SAVE_PATH

    device = config.DEVICE
    print(f"[Train] Device: {device}")

    # Load labels & splits
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    splits = np.load(os.path.join(data_dir, "splits.npy"))
    images_dir = os.path.join(data_dir, "images")

    # Normalise labels to [-1, 1] for better training
    # pitch ∈ [-90, 90] → /90;  roll, yaw ∈ [-180, 180] → /180
    label_scales = np.array([90.0, 180.0, 180.0], dtype=np.float32)
    labels_norm = labels / label_scales

    train_ds = StarImageDataset(images_dir, labels_norm, splits == 0)
    val_ds = StarImageDataset(images_dir, labels_norm, splits == 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"[Train] Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # Model
    model = StarCNN(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "val_mae_deg": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_losses = []
        for imgs, labels_batch in train_loader:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        val_errors_deg = []
        with torch.no_grad():
            for imgs, labels_batch in val_loader:
                imgs = imgs.to(device)
                labels_batch = labels_batch.to(device)

                preds = model(imgs)
                loss = criterion(preds, labels_batch)
                val_losses.append(loss.item())

                # Convert back to degrees for interpretable error
                preds_deg = preds.cpu().numpy() * label_scales
                true_deg = labels_batch.cpu().numpy() * label_scales
                mae = np.mean(np.abs(preds_deg - true_deg), axis=0)
                val_errors_deg.append(mae)

        avg_val = np.mean(val_losses)
        mae_deg = np.mean(val_errors_deg, axis=0)

        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_mae_deg"].append(mae_deg.tolist())

        elapsed = time.time() - t0

        print(f"  Epoch {epoch:>3}/{epochs}  "
              f"train={avg_train:.6f}  val={avg_val:.6f}  "
              f"MAE(°) P={mae_deg[0]:.2f} R={mae_deg[1]:.2f} Y={mae_deg[2]:.2f}  "
              f"({elapsed:.1f}s)")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "label_scales": label_scales.tolist(),
            }, save_path)
            print(f"  ✓ Saved best model (val_loss={avg_val:.6f})")

    print(f"\n[Train] ✓ Training complete. Best val loss: {best_val_loss:.6f}")
    print(f"[Train] Model saved to: {save_path}")
    return history


# ─────────────────────── CLI ─────────────────────── #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Star CNN")
    parser.add_argument("-e", "--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("-b", "--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
