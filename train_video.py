"""
Training script for the temporal deepfake detection model.

Fine-tunes the Transformer/LSTM temporal head on video frame sequences
while keeping the InceptionResnetV1 backbone frozen.

Usage:
    python train_video.py \
        --train_dir ./data/video_frames/train \
        --val_dir ./data/video_frames/val \
        --epochs 30 \
        --seq_len 16 \
        --temporal_head transformer

Expected data layout (created by preprocess_videos.py):
    data/video_frames/
        train/
            real/
                video001/  (frame_0000.jpg, frame_0001.jpg, ...)
                video002/
            fake/
                video001/
                video002/
        val/
            real/
                ...
            fake/
                ...
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np

from video_dataset import create_video_dataloaders
from temporal_model import TemporalDeepfakeModel


# ============================================================
# Training loop for one epoch
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train the temporal model for one epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (frames, labels) in enumerate(loader):
        # frames: (B, T, 3, H, W), labels: (B,)
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(frames).squeeze(-1)  # (B,)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg = running_loss / num_batches
            print(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{len(loader)} — Loss: {avg:.4f}")

    return running_loss / max(num_batches, 1)


# ============================================================
# Validation loop
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the temporal model. Returns dict of metrics."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    num_batches = 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(frames).squeeze(-1)
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
    print(f"[TrainVideo] Using device: {device}")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    print(f"[TrainVideo] Loading video data...")
    train_loader, val_loader = create_video_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Model
    print(f"[TrainVideo] Building temporal model...")
    backbone_weights = args.backbone_weights
    if backbone_weights is None:
        # Try to use existing fine-tuned checkpoint as backbone
        default_ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
        if os.path.isfile(default_ckpt):
            backbone_weights = default_ckpt
            print(f"[TrainVideo] Using existing fine-tuned backbone: {default_ckpt}")

    model = TemporalDeepfakeModel(
        temporal_head=args.temporal_head,
        backbone_weights=backbone_weights,
        freeze_backbone=not args.unfreeze_backbone,
        device=device,
        seq_len=args.seq_len,
    )
    model = model.to(device)

    # Loss & Optimizer — only optimize trainable parameters
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    start_epoch = 1
    best_auc = 0.0
    patience_counter = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[Train] Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_auc = checkpoint['best_auc']
            # fast forward scheduler
            for _ in range(start_epoch - 1):
                scheduler.step()
        else:
            print(f"[Train] Checkpoint not found at {args.resume}, starting from scratch.")

    print(f"\n{'=' * 60}")
    print(f"Starting TEMPORAL model training for {args.epochs} epochs")
    print(f"  Temporal head:   {args.temporal_head}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Weight decay:    {args.weight_decay}")
    print(f"  Backbone frozen: {not args.unfreeze_backbone}")
    print(f"  Early stopping:  {args.patience} epochs")
    print(f"  Checkpoints:     {ckpt_dir}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
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

            best_path = ckpt_dir / 'best_temporal_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics,
                'temporal_head': args.temporal_head,
                'seq_len': args.seq_len,
            }, best_path)
            print(f"  ✓ New best temporal model saved (AUC: {best_auc:.4f}) → {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping.")
            break

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f'temporal_checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics,
                'temporal_head': args.temporal_head,
                'seq_len': args.seq_len,
            }, periodic_path)
            print(f"  Periodic checkpoint saved → {periodic_path}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"Temporal Training Complete!")
    print(f"  Best validation AUC: {best_auc:.4f}")
    print(f"  Best checkpoint:     {ckpt_dir / 'best_temporal_model.pth'}")
    print(f"{'=' * 60}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train temporal deepfake detection model on video frame sequences'
    )

    # Data
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training video frames (containing real/ and fake/ subdirs)')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation video frames (containing real/ and fake/ subdirs)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input frame size (default: 256)')
    parser.add_argument('--seq_len', type=int, default=16,
                        help='Number of frames per video sequence (default: 16)')

    # Model
    parser.add_argument('--temporal_head', type=str, default='transformer',
                        choices=['transformer', 'lstm'],
                        help='Temporal head architecture (default: transformer)')
    parser.add_argument('--backbone_weights', type=str, default=None,
                        help='Path to fine-tuned backbone checkpoint (default: auto-detect)')
    parser.add_argument('--unfreeze_backbone', action='store_true',
                        help='Unfreeze backbone for end-to-end fine-tuning (uses more memory)')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8, lower than image training due to memory)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience in epochs (default: 7)')

    # Checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints (default: ./checkpoints)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save periodic checkpoint every N epochs (default: 5)')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    args = parser.parse_args()
    main(args)
