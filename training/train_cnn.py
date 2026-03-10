import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.cnn_dataset_loader import CNNDrivingDataset
from models.cnn_regression_model import CNNRegressionModel

def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Coefficient of determination R²."""
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """Single train or validation pass. Returns (avg_loss, avg_mae, r2)."""
    model.train() if train else model.eval()

    total_loss, total_mae = 0.0, 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, angles in loader:
            images  = images.to(device)
            angles  = angles.to(device)

            preds = model(images)
            loss  = criterion(preds, angles)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(images)
            total_mae  += (preds - angles).abs().sum().item()
            all_preds.append(preds.detach().cpu())
            all_targets.append(angles.detach().cpu())

    n          = len(loader.dataset)
    avg_loss   = total_loss / n
    avg_mae    = total_mae  / n
    r2         = r2_score(
        torch.cat(all_preds).squeeze(),
        torch.cat(all_targets).squeeze()
    )
    return avg_loss, avg_mae, r2

# Main training function 

def train_cnn_model(
    data_dir:      str   = "dataset/data",
    save_path:     str   = "cnn_regression_model.pth",
    epochs:        int   = 30,
    batch_size:    int   = 64,
    lr:            float = 1e-3,
    train_ratio:   float = 0.8,
    patience:      int   = 7,       # early-stopping patience (epochs)
    seed:          int   = 42,
):

    # Reproducibility 
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== CNN Regression Training ===")
    print(f"Device : {device}")
    print(f"Epochs : {epochs}  |  Batch: {batch_size}  |  LR: {lr}\n")

    # Dataset 
    full_ds = CNNDrivingDataset(root_dir=data_dir, mode="train")
    train_ds, val_ds = full_ds.split(train_ratio=train_ratio, seed=seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}\n")

    # Model 
    model = CNNRegressionModel().to(device)
    print(f"Parameters    : {model.count_parameters():,}\n")

    # Optimiser & scheduler 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop 
    best_val_loss  = float("inf")
    patience_count = 0
    history        = []

    header = (f"{'Epoch':>6} | {'Train MSE':>10} {'Train MAE':>10} "
              f"{'Train R²':>9} | {'Val MSE':>9} {'Val MAE':>9} "
              f"{'Val R²':>8} | {'LR':>9} | {'Time':>6}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_mae, tr_r2 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_mae, vl_r2 = run_epoch(
            model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(vl_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6} | {tr_loss:>10.6f} {tr_mae:>10.6f} "
              f"{tr_r2:>9.4f} | {vl_loss:>9.6f} {vl_mae:>9.6f} "
              f"{vl_r2:>8.4f} | {current_lr:>9.2e} | {elapsed:>5.1f}s")

        history.append({
            "epoch": epoch,
            "train_mse": tr_loss, "train_mae": tr_mae, "train_r2": tr_r2,
            "val_mse":   vl_loss, "val_mae":   vl_mae, "val_r2":   vl_r2,
        })

        # Save best checkpoint 
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_mse":     vl_loss,
                "val_r2":      vl_r2,
            }, save_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    # Load best weights for return 
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"\nBest checkpoint → {save_path}")
    print(f"  Best Val MSE : {checkpoint['val_mse']:.6f}")
    print(f"  Best Val R²  : {checkpoint['val_r2']:.4f}")

    return model, history

if __name__ == "__main__":
    train_cnn_model()