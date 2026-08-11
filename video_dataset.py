"""
VideoDeepfakeDataset — PyTorch Dataset for temporal deepfake detection training.

Loads sequences of face frames per video, returning tensors of shape
(seq_len, 3, H, W) for training temporal models (Transformer / LSTM).

Expected directory structure (created by preprocess_videos.py):
    dataset_root/
        real/
            video001/
                frame_0000.jpg
                frame_0001.jpg
                ...
            video002/
                ...
        fake/
            video001/
                ...
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

# Reuse augmentations from the image dataset
from dataset import JPEGCompression, RandomDownscaleUpscale, RandomGaussianBlur


# ============================================================
# Image extensions
# ============================================================

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# ============================================================
# Temporal augmentation transforms
# ============================================================

def get_video_train_transforms(image_size=256):
    """Per-frame spatial augmentations for video training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        RandomGaussianBlur(p=0.2),
        JPEGCompression(quality_range=(40, 95)),
        transforms.ToTensor(),  # [0, 1]
    ])


def get_video_val_transforms(image_size=256):
    """Deterministic per-frame transforms for validation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
    ])


# ============================================================
# VideoDeepfakeDataset
# ============================================================

class VideoDeepfakeDataset(Dataset):
    """
    Loads sequences of face frames for temporal deepfake detection.

    Each sample is a video represented as a sequence of face frame images.
    Returns (frames_tensor, label) where frames_tensor has shape (seq_len, 3, H, W).

    Args:
        root_dir:    Path to dataset root containing 'real/' and 'fake/' subdirs,
                     each with per-video subdirectories of frame images.
        seq_len:     Number of frames per sequence. Videos with fewer frames
                     are padded; videos with more are subsampled.
        transform:   Torchvision transform pipeline applied per frame.
        temporal_jitter: If True, add random offsets when sampling frames (training).
        max_videos:  Optional cap on total video count.
    """

    def __init__(
        self,
        root_dir: str,
        seq_len: int = 16,
        transform=None,
        temporal_jitter: bool = False,
        max_videos: int = None,
    ):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.transform = transform
        self.temporal_jitter = temporal_jitter

        # Each sample: (video_dir_path, label)
        self.samples = []

        real_dir = self.root_dir / 'real'
        fake_dir = self.root_dir / 'fake'

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Expected 'real/' and 'fake/' subdirectories inside '{root_dir}'.\n"
                f"  real_dir exists: {real_dir.exists()}\n"
                f"  fake_dir exists: {fake_dir.exists()}\n"
                f"Run preprocess_videos.py first to create the expected structure."
            )

        # Collect video directories (each subdir = one video's frames)
        for video_dir in sorted(real_dir.iterdir()):
            if video_dir.is_dir() and self._has_frames(video_dir):
                self.samples.append((str(video_dir), 0))  # 0 = Real

        for video_dir in sorted(fake_dir.iterdir()):
            if video_dir.is_dir() and self._has_frames(video_dir):
                self.samples.append((str(video_dir), 1))  # 1 = Fake

        random.shuffle(self.samples)

        if max_videos is not None:
            self.samples = self.samples[:max_videos]

        # Stats
        real_count = sum(1 for _, lbl in self.samples if lbl == 0)
        fake_count = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[VideoDeepfakeDataset] Loaded {len(self.samples)} videos "
              f"(real={real_count}, fake={fake_count}) from '{root_dir}'")

    def _has_frames(self, video_dir: Path) -> bool:
        """Check if a video directory has at least one frame image."""
        return any(
            f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in video_dir.iterdir()
        )

    def _get_frame_paths(self, video_dir: str) -> list:
        """Get sorted list of frame image paths for a video."""
        video_path = Path(video_dir)
        frames = [
            str(f) for f in sorted(video_path.iterdir())
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return frames

    def _sample_frames(self, frame_paths: list) -> list:
        """
        Sample exactly seq_len frames from the available frames.

        - If fewer frames available: repeat/pad to reach seq_len
        - If more frames available: evenly subsample
        - With temporal_jitter: add random offset to sampling positions
        """
        n_available = len(frame_paths)

        if n_available == 0:
            return []

        if n_available <= self.seq_len:
            # Pad by repeating the sequence
            sampled = frame_paths.copy()
            while len(sampled) < self.seq_len:
                sampled.append(frame_paths[len(sampled) % n_available])
            return sampled

        # Subsample: evenly spaced indices
        if self.temporal_jitter:
            # Add random offset for data augmentation
            max_offset = max(1, n_available // self.seq_len // 2)
            offset = random.randint(0, max_offset)
            indices = np.linspace(offset, n_available - 1, num=self.seq_len, dtype=int)
        else:
            indices = np.linspace(0, n_available - 1, num=self.seq_len, dtype=int)

        return [frame_paths[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_paths = self._get_frame_paths(video_dir)
        sampled_paths = self._sample_frames(frame_paths)

        frames = []
        for fp in sampled_paths:
            try:
                img = Image.open(fp).convert('RGB')
            except Exception as e:
                print(f"[VideoDeepfakeDataset] Error loading '{fp}': {e}")
                img = Image.new('RGB', (256, 256), (0, 0, 0))

            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            frames.append(img)

        # Handle edge case: no frames loaded
        if len(frames) == 0:
            frames = [torch.zeros(3, 256, 256) for _ in range(self.seq_len)]

        # Stack to (seq_len, 3, H, W)
        frames_tensor = torch.stack(frames, dim=0)

        # Normalize from [0, 1] to [-1, 1] (matching InceptionResnetV1 input)
        frames_tensor = (frames_tensor - 0.5) / 0.5

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return frames_tensor, label_tensor


# ============================================================
# Utility: Create video data loaders
# ============================================================

def create_video_dataloaders(
    train_dir: str,
    val_dir: str,
    seq_len: int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 256,
):
    """
    Create training and validation DataLoaders for video sequences.

    Args:
        train_dir:   Path to training video frames.
        val_dir:     Path to validation video frames.
        seq_len:     Number of frames per sequence.
        batch_size:  Batch size (lower than image training due to memory).
        num_workers: Parallel data loading workers.
        image_size:  Target frame size (square).

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = VideoDeepfakeDataset(
        root_dir=train_dir,
        seq_len=seq_len,
        transform=get_video_train_transforms(image_size),
        temporal_jitter=True,
    )
    val_dataset = VideoDeepfakeDataset(
        root_dir=val_dir,
        seq_len=seq_len,
        transform=get_video_val_transforms(image_size),
        temporal_jitter=False,
    )

    # Weighted sampler for class balance
    labels = [lbl for _, lbl in train_dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
    # Avoid division by zero if a class is missing
    weights = [
        1.0 / class_counts[lbl] if class_counts[lbl] > 0 else 0.0
        for lbl in labels
    ]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
