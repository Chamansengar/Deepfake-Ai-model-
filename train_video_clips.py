"""
Training script for temporal deepfake detection directly on raw video clips.

End-to-end pipeline: reads video files (.mp4, .avi, etc.), extracts faces
on the fly (with optional disk caching), and trains the Transformer/LSTM
temporal head on frame sequences.

Usage (single directory, auto-split):
    python train_video_clips.py \
        --video_dir ./data/videos \
        --val_split 0.2 \
        --epochs 30 \
        --seq_len 16 \
        --temporal_head transformer

Usage (separate directories):
    python train_video_clips.py \
        --train_video_dir ./data/videos_train \
        --val_video_dir ./data/videos_val \
        --epochs 30

Expected data layout:
    data/videos/
        real/
            video001.mp4
            video002.avi
            ...
        fake/
            video001.mp4
            video002.avi
            ...

The script will:
    1. Discover all video files in real/ and fake/ subdirectories
    2. Auto-split into train/val (or use pre-split directories)
    3. Extract face frames from each video on the fly (with optional caching)
    4. Train the TemporalDeepfakeModel (InceptionResnetV1 backbone + temporal head)
    5. Track AUC, accuracy, precision, recall on the validation set
    6. Save the best checkpoint (by val AUC) to checkpoints/
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np

from video_clip_dataset import create_video_clip_dataloaders
from temporal_model import TemporalDeepfakeModel


# ============================================================
# Training loop for one epoch
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    use_amp=False, scaler=None, trainable_params=None):
    """Train the temporal model for one epoch on video clips. Returns average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (frames, labels) in enumerate(loader):
        # frames: (B, T, 3, H, W), labels: (B,)
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(frames).squeeze(-1)  # (B,)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames).squeeze(-1)  # (B,)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
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
    print(f"[TrainVideoClips] Using device: {device}")

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    print(f"[TrainVideoClips] Loading video clips...")
    train_loader, val_loader = create_video_clip_dataloaders(
        video_dir=args.video_dir,
        train_video_dir=args.train_video_dir,
        val_video_dir=args.val_video_dir,
        val_split=args.val_split,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        skip_face_detection=args.skip_face_detection,
        cache_dir=args.cache_dir,
        clips_per_video=args.clips_per_video,
        num_extract_frames=args.num_extract_frames,
        max_videos=args.max_videos,
        seed=args.seed,
    )

    # Model
    print(f"[TrainVideoClips] Building temporal model...")
    backbone_weights = args.backbone_weights
    if backbone_weights is None:
        # Try to use existing fine-tuned checkpoint as backbone
        default_ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
        if os.path.isfile(default_ckpt):
            backbone_weights = default_ckpt
            print(f"[TrainVideoClips] Using existing fine-tuned backbone: {default_ckpt}")

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

    # Mixed precision
    use_amp = args.amp and device.startswith('cuda')
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print(f"[TrainVideoClips] Using automatic mixed precision (AMP)")

    # Training loop
    start_epoch = 1
    best_auc = 0.0
    patience_counter = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[TrainVideoClips] Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_auc = checkpoint['best_auc']
            # fast forward scheduler
            for _ in range(start_epoch - 1):
                scheduler.step()
        else:
            print(f"[TrainVideoClips] Checkpoint not found at {args.resume}, starting from scratch.")

    print(f"\n{'=' * 64}")
    print(f"Starting VIDEO CLIPS training for {args.epochs} epochs")
    print(f"  Pipeline:          Raw video clips → face extraction → temporal model")
    print(f"  Temporal head:     {args.temporal_head}")
    print(f"  Sequence length:   {args.seq_len}")
    print(f"  Extract frames:    {args.num_extract_frames} per video")
    print(f"  Clips per video:   {args.clips_per_video}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Weight decay:      {args.weight_decay}")
    print(f"  Backbone frozen:   {not args.unfreeze_backbone}")
    print(f"  Face detection:    {'OFF (raw frames)' if args.skip_face_detection else 'ON (MTCNN)'}")
    print(f"  Face cache:        {args.cache_dir or 'DISABLED'}")
    print(f"  Mixed precision:   {'ON' if use_amp else 'OFF'}")
    print(f"  Early stopping:    {args.patience} epochs")
    print(f"  Checkpoints:       {ckpt_dir}")
    print(f"{'=' * 64}\n")

    total_train_time = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_amp=use_amp, scaler=scaler, trainable_params=trainable_params,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        # Estimate remaining time
        epochs_remaining = args.epochs - epoch
        avg_epoch_time = total_train_time / (epoch - start_epoch + 1)
        eta = avg_epoch_time * epochs_remaining

        # Print epoch summary
        print(
            f"\n[Epoch {epoch}/{args.epochs}] ({epoch_time:.1f}s, ETA: {eta / 60:.1f}min) "
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

            best_path = ckpt_dir / 'best_temporal_clips_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics,
                'temporal_head': args.temporal_head,
                'seq_len': args.seq_len,
                'pipeline': 'video_clips',
            }, best_path)
            print(f"  ✓ New best model saved (AUC: {best_auc:.4f}) → {best_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs. Stopping.")
            break

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            periodic_path = ckpt_dir / f'temporal_clips_checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'val_metrics': val_metrics,
                'temporal_head': args.temporal_head,
                'seq_len': args.seq_len,
                'pipeline': 'video_clips',
            }, periodic_path)
            print(f"  Periodic checkpoint saved → {periodic_path}")

    # Final summary
    print(f"\n{'=' * 64}")
    print(f"Video Clips Training Complete!")
    print(f"  Total training time: {total_train_time / 60:.1f} minutes")
    print(f"  Best validation AUC: {best_auc:.4f}")
    print(f"  Best checkpoint:     {ckpt_dir / 'best_temporal_clips_model.pth'}")
    print(f"{'=' * 64}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train temporal deepfake detection model directly on raw video clips',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-split a single video directory (80/20 train/val):
  python train_video_clips.py --video_dir ./data/videos --epochs 30

  # Use separate train/val directories:
  python train_video_clips.py \\
      --train_video_dir ./data/videos_train \\
      --val_video_dir ./data/videos_val \\
      --epochs 30

  # Enable face caching for faster subsequent epochs:
  python train_video_clips.py --video_dir ./data/videos --cache_dir ./data/face_cache

  # Quick experiment with 5 videos per class:
  python train_video_clips.py --video_dir ./data/videos --max_videos 5 --epochs 3
        """,
    )

    # Data source (mutually exclusive modes)
    data_group = parser.add_argument_group('Data Source (choose one mode)')
    data_group.add_argument('--video_dir', type=str, default=None,
                            help='Single directory with real/ and fake/ subdirs (auto-split mode)')
    data_group.add_argument('--train_video_dir', type=str, default=None,
                            help='Pre-split training video directory (separate dir mode)')
    data_group.add_argument('--val_video_dir', type=str, default=None,
                            help='Pre-split validation video directory (separate dir mode)')
    data_group.add_argument('--val_split', type=float, default=0.2,
                            help='Fraction for validation (auto-split mode only, default: 0.2)')
    data_group.add_argument('--seed', type=int, default=42,
                            help='Random seed for train/val split (default: 42)')

    # Video processing
    proc_group = parser.add_argument_group('Video Processing')
    proc_group.add_argument('--image_size', type=int, default=256,
                            help='Frame size (default: 256)')
    proc_group.add_argument('--seq_len', type=int, default=16,
                            help='Frames per training sequence (default: 16)')
    proc_group.add_argument('--num_extract_frames', type=int, default=32,
                            help='Total frames to extract per video (default: 32)')
    proc_group.add_argument('--clips_per_video', type=int, default=1,
                            help='Non-overlapping clips per video per epoch (default: 1)')
    proc_group.add_argument('--skip_face_detection', action='store_true',
                            help='Skip MTCNN face detection, use raw frames (faster but less accurate)')
    proc_group.add_argument('--cache_dir', type=str, default=None,
                            help='Directory to cache extracted faces (None = no caching)')
    proc_group.add_argument('--max_videos', type=int, default=None,
                            help='Max videos per class (for quick experiments)')

    # Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--temporal_head', type=str, default='transformer',
                             choices=['transformer', 'lstm'],
                             help='Temporal head architecture (default: transformer)')
    model_group.add_argument('--backbone_weights', type=str, default=None,
                             help='Path to fine-tuned backbone checkpoint (default: auto-detect)')
    model_group.add_argument('--unfreeze_backbone', action='store_true',
                             help='Unfreeze backbone for end-to-end fine-tuning (uses more memory)')

    # Training
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=30,
                             help='Number of training epochs (default: 30)')
    train_group.add_argument('--batch_size', type=int, default=4,
                             help='Batch size (default: 4, lower than image training)')
    train_group.add_argument('--lr', type=float, default=5e-4,
                             help='Initial learning rate (default: 5e-4)')
    train_group.add_argument('--weight_decay', type=float, default=1e-4,
                             help='Weight decay for AdamW (default: 1e-4)')
    train_group.add_argument('--patience', type=int, default=7,
                             help='Early stopping patience in epochs (default: 7)')
    train_group.add_argument('--amp', action='store_true',
                             help='Enable automatic mixed precision (AMP) training')

    # Checkpoints
    ckpt_group = parser.add_argument_group('Checkpoints')
    ckpt_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                            help='Directory to save model checkpoints (default: ./checkpoints)')
    ckpt_group.add_argument('--resume', type=str, default='',
                            help='Path to checkpoint to resume training from')
    ckpt_group.add_argument('--save_every', type=int, default=5,
                            help='Save periodic checkpoint every N epochs (default: 5)')

    # DataLoader
    loader_group = parser.add_argument_group('DataLoader')
    loader_group.add_argument('--num_workers', type=int, default=4,
                              help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    # Validate arguments
    if args.video_dir is None and (args.train_video_dir is None or args.val_video_dir is None):
        parser.error(
            "Provide either --video_dir (auto-split) or both "
            "--train_video_dir and --val_video_dir (separate dirs)."
        )

    main(args)
