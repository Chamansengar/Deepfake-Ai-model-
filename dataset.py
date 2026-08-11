"""
DeepfakeDataset — PyTorch Dataset for deepfake detection training.

Expected directory structure:
    dataset_root/
        real/
            img001.jpg
            img002.png
            ...
        fake/
            img001.jpg
            img002.png
            ...

Supports both pre-cropped face images and full images (MTCNN will crop).
"""

import os
import io
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
from facenet_pytorch import MTCNN


# ============================================================
# Custom Augmentation Transforms
# ============================================================

class JPEGCompression:
    """Simulate JPEG compression artifacts (common in shared deepfakes)."""

    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class RandomDownscaleUpscale:
    """Simulate resolution changes from re-encoding / sharing."""

    def __init__(self, scale_range=(0.4, 0.9)):
        self.scale_range = scale_range

    def __call__(self, img):
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small_size = (max(int(w * scale), 16), max(int(h * scale), 16))
        img = img.resize(small_size, Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        return img


class RandomGaussianBlur:
    """Apply Gaussian blur with random radius."""

    def __init__(self, radius_range=(0.5, 2.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


# ============================================================
# Build augmentation pipelines
# ============================================================

def get_train_transforms(image_size=256):
    """Training augmentations — aggressive to improve generalization."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        RandomGaussianBlur(p=0.3),
        JPEGCompression(quality_range=(30, 95)),
        RandomDownscaleUpscale(scale_range=(0.5, 0.9)),
        transforms.ToTensor(),  # [0, 1]
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # small cutout expects tensor
    ])


def get_val_transforms(image_size=256):
    """Validation transforms — deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
    ])


# ============================================================
# DeepfakeDataset
# ============================================================

class DeepfakeDataset(Dataset):
    """
    Loads real/fake face images for deepfake detection training.

    Args:
        root_dir:       Path to dataset root containing 'real/' and 'fake/' subdirs.
        transform:      Torchvision transform pipeline to apply.
        use_mtcnn:      If True, detect and crop faces using MTCNN before transform.
                        Set False if images are already pre-cropped faces.
        max_samples:    Optional cap on total samples (useful for quick experiments).
        device:         Device for MTCNN ('cpu' recommended during data loading).
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        root_dir: str,
        transform=None,
        use_mtcnn: bool = False,
        max_samples: int = None,
        device: str = 'cpu',
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_mtcnn = use_mtcnn

        # Lazy-init MTCNN only if needed
        self.mtcnn = None
        if use_mtcnn:
            self.mtcnn = MTCNN(
                image_size=256,
                margin=40,
                select_largest=True,
                post_process=False,
                device=device,
            )

        # Collect file paths and labels
        self.samples = []  # list of (filepath, label)

        real_dir = self.root_dir / 'real'
        fake_dir = self.root_dir / 'fake'

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Expected 'real/' and 'fake/' subdirectories inside '{root_dir}'.\n"
                f"  real_dir exists: {real_dir.exists()}\n"
                f"  fake_dir exists: {fake_dir.exists()}"
            )

        for img_path in sorted(real_dir.iterdir()):
            if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.samples.append((str(img_path), 0))  # 0 = Real

        for img_path in sorted(fake_dir.iterdir()):
            if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.samples.append((str(img_path), 1))  # 1 = Fake

        # Shuffle to mix real/fake
        random.shuffle(self.samples)

        # Optional cap
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Stats
        real_count = sum(1 for _, lbl in self.samples if lbl == 0)
        fake_count = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[DeepfakeDataset] Loaded {len(self.samples)} samples "
              f"(real={real_count}, fake={fake_count}) from '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[DeepfakeDataset] Error loading '{img_path}': {e}")
            # Return a black image with the correct label as fallback
            img = Image.new('RGB', (256, 256), (0, 0, 0))

        # Optional MTCNN face crop
        if self.mtcnn is not None:
            face = self.mtcnn(img)
            if face is not None:
                # MTCNN returns tensor [3, H, W] in [0, 255]
                # Convert back to PIL for the transform pipeline
                face_np = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                img = Image.fromarray(face_np)
            # If no face detected, use the original image (it might be a tight crop already)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Normalize from [0, 1] to [-1, 1] (matching InceptionResnetV1 expected input)
        img = (img - 0.5) / 0.5

        label = torch.tensor(label, dtype=torch.float32)
        return img, label


# ============================================================
# Utility: Create balanced data loaders
# ============================================================

def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 256,
    use_mtcnn: bool = False,
):
    """
    Create training and validation DataLoaders.

    Args:
        train_dir: Path to training data (containing real/ and fake/).
        val_dir:   Path to validation data (containing real/ and fake/).
        batch_size: Batch size for both loaders.
        num_workers: Number of parallel data loading workers.
        image_size: Target image size (square).
        use_mtcnn: Whether to run MTCNN face detection during loading.

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler

    train_dataset = DeepfakeDataset(
        root_dir=train_dir,
        transform=get_train_transforms(image_size),
        use_mtcnn=use_mtcnn,
    )
    val_dataset = DeepfakeDataset(
        root_dir=val_dir,
        transform=get_val_transforms(image_size),
        use_mtcnn=use_mtcnn,
    )

    # Weighted sampler to handle class imbalance
    labels = [lbl for _, lbl in train_dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[lbl] for lbl in labels]
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
