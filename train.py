"""
Training script for fine-tuning InceptionResnetV1 on deepfake detection.

Usage:
    python train.py --train_dir ./data/train --val_dir ./data/val --epochs 20

Expected data layout:
    data/
        train/
            real/   (images)
            fake/   (images)
        val/
            real/   (images)
            fake/   (images)

The script will:
    1. Load InceptionResnetV1 pretrained on VGGFace2
    2. Freeze early layers, unfreeze last blocks + classifier
    3. Train with BCEWithLogitsLoss + class-balanced sampling
    4. Track AUC, accuracy, precision, recall on validation set
    5. Save the best checkpoint (by val AUC) to checkpoints/
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np

from dataset import create_dataloaders


# ============================================================
# Model setup with selective unfreezing
# ============================================================

def build_model(device='cpu', unfreeze_blocks=('block8', 'last_linear', 'last_bn', 'logits')):
    """
    Load InceptionResnetV1 pretrained on VGGFace2 and selectively unfreeze
    the last few blocks for fine-tuning on deepfake detection.

    Args:
        device: Target device.
        unfreeze_blocks: Tuple of module name prefixes to unfreeze.

    Returns:
        model on the specified device.
    """
    model = InceptionResnetV1(
        pretrained='vggface2',
        classify=True,
        num_classes=1,
        device=device,
    )

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Selectively unfreeze
    unfrozen_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if any(name.startswith(block) for block in unfreeze_blocks):
            param.requires_grad = True
            unfrozen_params += param.numel()

    print(f"[Model] Total parameters:    {total_params:,}")
    print(f"[Model] Trainable parameters: {unfrozen_params:,} "
          f"({unfrozen_params / total_params * 100:.1f}%)")

    return model.to(device)


# ============================================================
# Training loop for one epoch
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch, return average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images).squeeze(-1)  # (B,)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            avg = running_loss / num_batches
            print(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{len(loader)} — Loss: {avg:.4f}")

    return running_loss / max(num_batches, 1)


# ============================================================
# Validation loop
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Validate the model. Returns dict with loss, accuracy, AUC, precision, recall.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    num_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images).squeeze(-1)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        num_batches += 1

        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    avg_loss = running_loss / max(num_batches, 1)

    # Handle edge case where only one class is present in batch
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'auc': auc,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
    }
    return metrics


# ============================================================
# Main training routine
# ============================================================

def main(args):
    # Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[Train] Using device: {device}")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    print(f"[Train] Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_mtcnn=args.use_mtcnn,
    )

    # Model
    print(f"[Train] Building model...")
    model = build_model(device=device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    best_auc = 0.0
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Weight decay:   {args.weight_decay}")
    print(f"  Early stopping: {args.patience} epochs")
    print(f"  Checkpoints:    {ckpt_dir}")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(
            f"\n[Epoch {epoch}/{args.epochs}] ({epoch_time:.1f}s) "
            f"lr={current_lr:.2e}\n"
            f"  Train Loss: {train_loss:.4f}\n"
            f"  Val Loss:   {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | "
            f"Prec: {val_metrics['precision']:.4f} | "
            f"Rec: {val_metrics['recall']:.4f}"
        )

        # Save best model by AUC
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0

            best_path = ckpt_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics,
            }, best_path)
            print(f"  ✓ New best model saved (AUC: {best_auc:.4f}) → {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping.")
            break

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics,
            }, periodic_path)
            print(f"  Periodic checkpoint saved → {periodic_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Best validation AUC: {best_auc:.4f}")
    print(f"  Best checkpoint:     {ckpt_dir / 'best_model.pth'}")
    print(f"{'=' * 60}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune InceptionResnetV1 for deepfake detection')

    # Data
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data (containing real/ and fake/ subdirs)')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data (containing real/ and fake/ subdirs)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size (default: 256)')
    parser.add_argument('--use_mtcnn', action='store_true',
                        help='Run MTCNN face detection during data loading (use if images are not pre-cropped)')

    # Training
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience in epochs (default: 5)')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save periodic checkpoint every N epochs (default: 5)')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    args = parser.parse_args()
    main(args)
