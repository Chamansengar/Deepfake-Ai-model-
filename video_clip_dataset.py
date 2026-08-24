"""
VideoClipDataset — PyTorch Dataset for training directly on raw video clip files.

Reads video files (.mp4, .avi, etc.) at training time, extracts face frames
on the fly using MTCNN, and returns frame sequences for temporal deepfake
detection models. Supports optional disk caching for faster subsequent epochs.

Expected directory structure:
    dataset_root/
        real/
            video001.mp4
            video002.avi
            ...
        fake/
            video001.mp4
            video002.avi
            ...

Usage:
    from video_clip_dataset import create_video_clip_dataloaders

    train_loader, val_loader = create_video_clip_dataloaders(
        video_dir='./data/videos',
        val_split=0.2,
        seq_len=16,
        batch_size=4,
    )
"""

import os
import random
import hashlib
import json
from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN

# Reuse augmentations from the image dataset
from dataset import JPEGCompression, RandomGaussianBlur


# ============================================================
# Supported extensions
# ============================================================

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


# ============================================================
# Transforms
# ============================================================

def get_clip_train_transforms(image_size=256):
    """Per-frame spatial augmentations for video clip training."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        RandomGaussianBlur(p=0.2),
        JPEGCompression(quality_range=(40, 95)),
        transforms.ToTensor(),  # [0, 1]
    ])


def get_clip_val_transforms(image_size=256):
    """Deterministic per-frame transforms for validation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
    ])


# ============================================================
# Face cache helpers
# ============================================================

def _video_cache_key(video_path: str, num_frames: int, face_size: int) -> str:
    """Generate a deterministic cache key for a video's extracted faces."""
    # Use file path + size + mtime for cache invalidation
    stat = os.stat(video_path)
    raw = f"{video_path}|{stat.st_size}|{stat.st_mtime}|{num_frames}|{face_size}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cached_faces(cache_dir: str, cache_key: str) -> list:
    """Load cached face frames from disk. Returns list of PIL Images or None."""
    cache_path = Path(cache_dir) / cache_key
    if not cache_path.exists():
        return None

    meta_path = cache_path / "meta.json"
    if not meta_path.exists():
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        num_faces = meta.get('num_faces', 0)
        if num_faces == 0:
            return []

        faces = []
        for i in range(num_faces):
            img_path = cache_path / f"face_{i:04d}.jpg"
            if img_path.exists():
                faces.append(Image.open(str(img_path)).convert('RGB'))
        return faces if len(faces) == num_faces else None
    except Exception:
        return None


def _save_cached_faces(cache_dir: str, cache_key: str, faces: list):
    """Save extracted face frames to disk cache."""
    cache_path = Path(cache_dir) / cache_key
    cache_path.mkdir(parents=True, exist_ok=True)

    for i, face_img in enumerate(faces):
        img_path = cache_path / f"face_{i:04d}.jpg"
        face_img.save(str(img_path), quality=95)

    meta_path = cache_path / "meta.json"
    with open(meta_path, 'w') as f:
        json.dump({'num_faces': len(faces)}, f)


# ============================================================
# Video frame extraction
# ============================================================

def extract_faces_from_clip(
    video_path: str,
    num_frames: int = 16,
    face_size: int = 256,
    margin: int = 40,
    skip_face_detection: bool = False,
) -> list:
    """
    Extract face-cropped frames from a video file.

    Args:
        video_path:           Path to video file.
        num_frames:           Number of frames to sample.
        face_size:            Output face crop size (square).
        margin:               MTCNN margin around detected face.
        skip_face_detection:  If True, return raw frames without face detection.

    Returns:
        List of PIL Images (face crops or raw frames).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample frame indices evenly across the video, skipping edges
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        start = max(1, int(total_frames * 0.02))
        end = min(total_frames - 1, int(total_frames * 0.98))
        if end <= start:
            start, end = 0, total_frames - 1
        frame_indices = np.linspace(start, end, num=num_frames, dtype=int).tolist()
        frame_indices = sorted(set(frame_indices))

    # Initialize MTCNN if needed
    mtcnn = None
    if not skip_face_detection:
        mtcnn = MTCNN(
            image_size=face_size,
            margin=margin,
            select_largest=True,
            post_process=False,
            device='cpu',
            min_face_size=60,
        )

    faces = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        if skip_face_detection:
            # Use raw frame resized to face_size
            img_pil = img_pil.resize((face_size, face_size), Image.LANCZOS)
            faces.append(img_pil)
        else:
            face_tensor = mtcnn(img_pil)
            if face_tensor is not None:
                # MTCNN returns tensor [3, H, W] in [0, 255]
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                face_img = Image.fromarray(face_np)
                faces.append(face_img)

    cap.release()
    return faces


# ============================================================
# VideoClipDataset
# ============================================================

class VideoClipDataset(Dataset):
    """
    PyTorch Dataset that reads raw video clip files for temporal deepfake detection.

    Each sample is a video file. On __getitem__, it extracts face frames
    on the fly (or from disk cache), applies transforms, and returns a
    frame sequence tensor.

    Args:
        video_paths:          List of (video_path, label) tuples.
        seq_len:              Number of frames per sequence.
        transform:            Torchvision transform pipeline per frame.
        temporal_jitter:      Add random temporal offsets during sampling.
        face_size:            Face crop size for MTCNN extraction.
        skip_face_detection:  Skip MTCNN and use raw frames.
        cache_dir:            Directory for disk-caching extracted faces.
                              None = no caching (re-extract every epoch).
        clips_per_video:      Number of non-overlapping clips per video per epoch.
        num_extract_frames:   How many frames to extract from each video
                              (before selecting seq_len from them).
    """

    def __init__(
        self,
        video_paths: list,
        seq_len: int = 16,
        transform=None,
        temporal_jitter: bool = False,
        face_size: int = 256,
        skip_face_detection: bool = False,
        cache_dir: str = None,
        clips_per_video: int = 1,
        num_extract_frames: int = 32,
    ):
        self.video_paths = video_paths
        self.seq_len = seq_len
        self.transform = transform
        self.temporal_jitter = temporal_jitter
        self.face_size = face_size
        self.skip_face_detection = skip_face_detection
        self.cache_dir = cache_dir
        self.clips_per_video = max(1, clips_per_video)
        self.num_extract_frames = max(num_extract_frames, seq_len)

        # Build the effective sample list: each video can produce multiple clips
        self.samples = []  # (video_path, label, clip_index)
        for video_path, label in self.video_paths:
            for clip_idx in range(self.clips_per_video):
                self.samples.append((video_path, label, clip_idx))

        # Stats
        real_count = sum(1 for _, lbl in self.video_paths if lbl == 0)
        fake_count = sum(1 for _, lbl in self.video_paths if lbl == 1)
        print(f"[VideoClipDataset] {len(self.video_paths)} videos "
              f"(real={real_count}, fake={fake_count}) → "
              f"{len(self.samples)} clips (×{self.clips_per_video})")
        if self.cache_dir:
            print(f"[VideoClipDataset] Face cache: {self.cache_dir}")

    def _get_faces(self, video_path: str) -> list:
        """Get face frames for a video, using cache if available."""
        cache_key = None

        # Try loading from cache
        if self.cache_dir:
            cache_key = _video_cache_key(video_path, self.num_extract_frames, self.face_size)
            cached = _load_cached_faces(self.cache_dir, cache_key)
            if cached is not None:
                return cached

        # Extract faces from the video file
        faces = extract_faces_from_clip(
            video_path=video_path,
            num_frames=self.num_extract_frames,
            face_size=self.face_size,
            skip_face_detection=self.skip_face_detection,
        )

        # Save to cache
        if self.cache_dir and cache_key and len(faces) > 0:
            try:
                _save_cached_faces(self.cache_dir, cache_key, faces)
            except Exception as e:
                print(f"[VideoClipDataset] Cache write error: {e}")

        return faces

    def _sample_clip_frames(self, faces: list, clip_index: int) -> list:
        """
        Sample seq_len frames from extracted faces for a specific clip index.

        For multi-clip mode, divides the face list into non-overlapping segments.
        Applies temporal jitter if enabled.
        """
        n = len(faces)
        if n == 0:
            return []

        # Determine the segment for this clip
        if self.clips_per_video > 1 and n >= self.clips_per_video * self.seq_len:
            segment_size = n // self.clips_per_video
            start = clip_index * segment_size
            end = start + segment_size
            segment_faces = faces[start:end]
        else:
            segment_faces = faces

        n_seg = len(segment_faces)

        # Pad if fewer frames than seq_len
        if n_seg <= self.seq_len:
            sampled = list(segment_faces)
            while len(sampled) < self.seq_len:
                sampled.append(segment_faces[len(sampled) % n_seg])
            return sampled

        # Subsample: evenly spaced with optional jitter
        if self.temporal_jitter:
            max_offset = max(1, n_seg // self.seq_len // 2)
            offset = random.randint(0, max_offset)
            indices = np.linspace(offset, n_seg - 1, num=self.seq_len, dtype=int)
        else:
            indices = np.linspace(0, n_seg - 1, num=self.seq_len, dtype=int)

        return [segment_faces[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, clip_index = self.samples[idx]

        # Extract / load cached faces
        faces = self._get_faces(video_path)

        # Sample a clip's worth of frames
        clip_frames = self._sample_clip_frames(faces, clip_index)

        # Apply transforms to each frame
        frame_tensors = []
        for face_img in clip_frames:
            try:
                if self.transform:
                    tensor = self.transform(face_img)
                else:
                    tensor = transforms.ToTensor()(face_img)
                frame_tensors.append(tensor)
            except Exception as e:
                print(f"[VideoClipDataset] Transform error: {e}")
                frame_tensors.append(torch.zeros(3, 256, 256))

        # Handle edge case: no frames at all
        if len(frame_tensors) == 0:
            frame_tensors = [torch.zeros(3, 256, 256) for _ in range(self.seq_len)]

        # Pad to seq_len if we still don't have enough
        n_existing = len(frame_tensors)
        i = n_existing
        while len(frame_tensors) < self.seq_len:
            frame_tensors.append(frame_tensors[i % n_existing])
            i += 1
        frame_tensors = frame_tensors[:self.seq_len]

        # Stack: (seq_len, 3, H, W)
        frames_tensor = torch.stack(frame_tensors, dim=0)

        # Normalize from [0, 1] to [-1, 1] (matching InceptionResnetV1 input)
        frames_tensor = (frames_tensor - 0.5) / 0.5

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return frames_tensor, label_tensor


# ============================================================
# Utility: Collect video files from directory
# ============================================================

def collect_video_files(root_dir: str) -> list:
    """
    Collect video files from a directory with real/ and fake/ subdirs.

    Args:
        root_dir: Path containing real/ and fake/ subdirectories with video files.

    Returns:
        List of (video_path, label) tuples. label=0 for real, label=1 for fake.
    """
    root = Path(root_dir)
    real_dir = root / 'real'
    fake_dir = root / 'fake'

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Expected 'real/' and 'fake/' subdirectories inside '{root_dir}'.\n"
            f"  real_dir exists: {real_dir.exists()}\n"
            f"  fake_dir exists: {fake_dir.exists()}"
        )

    videos = []

    for video_file in sorted(real_dir.iterdir()):
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append((str(video_file), 0))  # 0 = Real

    for video_file in sorted(fake_dir.iterdir()):
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append((str(video_file), 1))  # 1 = Fake

    return videos


# ============================================================
# Utility: Train/val split
# ============================================================

def split_videos(videos: list, val_split: float = 0.2, seed: int = 42) -> tuple:
    """
    Split video list into train and validation sets.

    Args:
        videos:    List of (video_path, label) tuples.
        val_split: Fraction for validation.
        seed:      Random seed for reproducibility.

    Returns:
        (train_videos, val_videos)
    """
    # Split per class for stratified split
    real_vids = [(p, l) for p, l in videos if l == 0]
    fake_vids = [(p, l) for p, l in videos if l == 1]

    rng = random.Random(seed)
    rng.shuffle(real_vids)
    rng.shuffle(fake_vids)

    real_split = int(len(real_vids) * (1 - val_split))
    fake_split = int(len(fake_vids) * (1 - val_split))

    train = real_vids[:real_split] + fake_vids[:fake_split]
    val = real_vids[real_split:] + fake_vids[fake_split:]

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ============================================================
# Utility: Create data loaders
# ============================================================

def create_video_clip_dataloaders(
    video_dir: str = None,
    train_video_dir: str = None,
    val_video_dir: str = None,
    val_split: float = 0.2,
    seq_len: int = 16,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 256,
    skip_face_detection: bool = False,
    cache_dir: str = None,
    clips_per_video: int = 1,
    num_extract_frames: int = 32,
    max_videos: int = None,
    seed: int = 42,
):
    """
    Create training and validation DataLoaders for raw video clips.

    Supports two modes:
    1. Single directory + auto-split: Pass video_dir with val_split.
    2. Separate directories: Pass train_video_dir and val_video_dir.

    Args:
        video_dir:            Single directory with real/ and fake/ subdirs (auto-split mode).
        train_video_dir:      Pre-split training directory (separate dir mode).
        val_video_dir:        Pre-split validation directory (separate dir mode).
        val_split:            Fraction for validation (only used with video_dir).
        seq_len:              Frames per sequence.
        batch_size:           Batch size.
        num_workers:          Data loading workers.
        image_size:           Target frame size.
        skip_face_detection:  Skip MTCNN and use raw frames.
        cache_dir:            Directory for face extraction cache.
        clips_per_video:      Clips to sample per video per epoch.
        num_extract_frames:   Total frames to extract from each video.
        max_videos:           Cap on total videos per class (for debugging).
        seed:                 Random seed.

    Returns:
        (train_loader, val_loader)
    """
    if video_dir is not None:
        # Auto-split mode
        all_videos = collect_video_files(video_dir)
        if max_videos is not None:
            real_vids = [(p, l) for p, l in all_videos if l == 0][:max_videos]
            fake_vids = [(p, l) for p, l in all_videos if l == 1][:max_videos]
            all_videos = real_vids + fake_vids
        train_videos, val_videos = split_videos(all_videos, val_split, seed)
    elif train_video_dir is not None and val_video_dir is not None:
        # Separate directory mode
        train_videos = collect_video_files(train_video_dir)
        val_videos = collect_video_files(val_video_dir)
        if max_videos is not None:
            train_real = [(p, l) for p, l in train_videos if l == 0][:max_videos]
            train_fake = [(p, l) for p, l in train_videos if l == 1][:max_videos]
            train_videos = train_real + train_fake
            val_real = [(p, l) for p, l in val_videos if l == 0][:max_videos]
            val_fake = [(p, l) for p, l in val_videos if l == 1][:max_videos]
            val_videos = val_real + val_fake
    else:
        raise ValueError(
            "Provide either 'video_dir' (auto-split) or both "
            "'train_video_dir' and 'val_video_dir' (separate dirs)."
        )

    print(f"[VideoClipDataLoaders] Train: {len(train_videos)} videos, "
          f"Val: {len(val_videos)} videos")

    # Cache subdirs per split to avoid collisions
    train_cache = os.path.join(cache_dir, 'train') if cache_dir else None
    val_cache = os.path.join(cache_dir, 'val') if cache_dir else None

    train_dataset = VideoClipDataset(
        video_paths=train_videos,
        seq_len=seq_len,
        transform=get_clip_train_transforms(image_size),
        temporal_jitter=True,
        face_size=image_size,
        skip_face_detection=skip_face_detection,
        cache_dir=train_cache,
        clips_per_video=clips_per_video,
        num_extract_frames=num_extract_frames,
    )
    val_dataset = VideoClipDataset(
        video_paths=val_videos,
        seq_len=seq_len,
        transform=get_clip_val_transforms(image_size),
        temporal_jitter=False,
        face_size=image_size,
        skip_face_detection=skip_face_detection,
        cache_dir=val_cache,
        clips_per_video=1,  # Single clip for deterministic validation
        num_extract_frames=num_extract_frames,
    )

    # Weighted sampler for class balance (based on underlying videos, not clips)
    labels = [lbl for _, lbl, _ in train_dataset.samples]
    class_counts = [labels.count(0), labels.count(1)]
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
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
