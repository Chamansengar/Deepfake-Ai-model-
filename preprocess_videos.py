"""
Video Preprocessing — Extract face frame sequences from video datasets.

Converts a directory of real/fake videos into organized face frame sequences
suitable for temporal deepfake detection training.

Usage:
    python preprocess_videos.py \
        --input_dir ./data/videos \
        --output_dir ./data/video_frames \
        --frames_per_video 16 \
        --val_split 0.2

Expected input structure:
    data/videos/
        real/
            video001.mp4
            video002.avi
            ...
        fake/
            video001.mp4
            video002.avi
            ...

Output structure:
    data/video_frames/
        train/
            real/
                video001/
                    frame_000.jpg
                    frame_001.jpg
                    ...
            fake/
                video001/
                    ...
        val/
            real/
                ...
            fake/
                ...
"""

import os
import argparse
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}


def get_video_files(directory: str) -> list:
    """Recursively find all video files in a directory."""
    video_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, f))
    return sorted(video_files)


def sample_frame_indices(total_frames: int, num_samples: int) -> list:
    """
    Sample frame indices evenly spaced across the video duration.
    Falls back to all available frames if fewer than requested.
    """
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))
    # Evenly space across the video, avoiding the very first/last frames
    # which may be black/transitional
    start = max(1, int(total_frames * 0.02))
    end = min(total_frames - 1, int(total_frames * 0.98))
    if end <= start:
        start, end = 0, total_frames - 1
    indices = np.linspace(start, end, num=num_samples, dtype=int).tolist()
    return sorted(set(indices))  # Remove duplicates


def extract_faces_from_video(
    video_path: str,
    output_dir: str,
    num_frames: int = 16,
    face_size: int = 256,
    margin: int = 40,
    min_face_size: int = 60,
) -> int:
    """
    Extract face crops from a single video.

    Args:
        video_path:  Path to the video file.
        output_dir:  Directory to save face frames into.
        num_frames:  Number of frames to sample from the video.
        face_size:   Output face crop size (square).
        margin:      MTCNN margin around detected face.
        min_face_size: Minimum face detection size.

    Returns:
        Number of face frames successfully extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        print(f"  [SKIP] Zero frames: {video_path}")
        return 0

    # Initialize MTCNN (per-process for multiprocessing safety)
    mtcnn = MTCNN(
        image_size=face_size,
        margin=margin,
        select_largest=True,
        post_process=False,
        device='cpu',  # CPU for preprocessing (GPU used during training)
        min_face_size=min_face_size,
    )

    frame_indices = sample_frame_indices(total_frames, num_frames)
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        # Detect and crop face
        face_tensor = mtcnn(img_pil)
        if face_tensor is None:
            continue

        # Convert tensor [3, H, W] (range [0, 255]) to PIL Image
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        face_img = Image.fromarray(face_np)

        # Save with zero-padded index for correct ordering
        save_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        face_img.save(save_path, quality=95)
        saved_count += 1

    cap.release()
    return saved_count


def process_single_video(args_tuple):
    """Wrapper for multiprocessing — unpacks arguments and calls extract_faces_from_video."""
    video_path, output_dir, num_frames, face_size, margin = args_tuple
    video_name = Path(video_path).stem
    video_output = os.path.join(output_dir, video_name)
    
    if os.path.exists(video_output):
        existing_frames = list(Path(video_output).glob("*.jpg"))
        if len(existing_frames) > 0:
            return video_path, video_name, len(existing_frames)
            
    count = extract_faces_from_video(
        video_path=video_path,
        output_dir=video_output,
        num_frames=num_frames,
        face_size=face_size,
        margin=margin,
    )
    return video_path, video_name, count


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    real_dir = input_dir / 'real'
    fake_dir = input_dir / 'fake'

    if not real_dir.exists() or not fake_dir.exists():
        print(f"ERROR: Expected 'real/' and 'fake/' subdirectories inside '{input_dir}'")
        print(f"  real/ exists: {real_dir.exists()}")
        print(f"  fake/ exists: {fake_dir.exists()}")
        return

    # Collect video files
    real_videos = get_video_files(str(real_dir))
    fake_videos = get_video_files(str(fake_dir))

    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")

    if args.max_videos:
        real_videos = real_videos[:args.max_videos]
        fake_videos = fake_videos[:args.max_videos]
        print(f"  (Capped to {args.max_videos} per class)")

    # Train/val split
    random.seed(args.seed)

    def split_videos(videos):
        random.shuffle(videos)
        split_idx = int(len(videos) * (1 - args.val_split))
        return videos[:split_idx], videos[split_idx:]

    real_train, real_val = split_videos(real_videos)
    fake_train, fake_val = split_videos(fake_videos)

    print(f"\nSplit:")
    print(f"  Train: {len(real_train)} real + {len(fake_train)} fake = {len(real_train) + len(fake_train)}")
    print(f"  Val:   {len(real_val)} real + {len(fake_val)} fake = {len(real_val) + len(fake_val)}")

    # Build processing tasks
    tasks = []
    splits = [
        (real_train, 'train', 'real'),
        (real_val,   'val',   'real'),
        (fake_train, 'train', 'fake'),
        (fake_val,   'val',   'fake'),
    ]

    for videos, split, label in splits:
        for video_path in videos:
            out = str(output_dir / split / label)
            tasks.append((video_path, out, args.frames_per_video, args.face_size, args.margin))

    print(f"\nProcessing {len(tasks)} videos with {args.workers} workers...")
    print(f"  Frames per video: {args.frames_per_video}")
    print(f"  Face size: {args.face_size}x{args.face_size}")
    print(f"  Margin: {args.margin}")
    print()

    # Process videos
    success_count = 0
    skip_count = 0
    total_frames = 0

    if args.workers <= 1:
        # Sequential processing (useful for debugging)
        for i, task in enumerate(tasks):
            video_path, video_name, count = process_single_video(task)
            total_frames += count
            if count > 0:
                success_count += 1
            else:
                skip_count += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                print(f"  Progress: {i + 1}/{len(tasks)} videos | "
                      f"{success_count} success, {skip_count} skipped, {total_frames} frames")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                video_path, video_name, count = future.result()
                total_frames += count
                if count > 0:
                    success_count += 1
                else:
                    skip_count += 1
                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    print(f"  Progress: {i + 1}/{len(tasks)} videos | "
                          f"{success_count} success, {skip_count} skipped, {total_frames} frames")

    # Remove empty video directories (videos where no face was detected)
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            label_dir = output_dir / split / label
            if label_dir.exists():
                for video_dir in label_dir.iterdir():
                    if video_dir.is_dir() and not any(video_dir.iterdir()):
                        video_dir.rmdir()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Preprocessing Complete!")
    print(f"  Videos processed: {success_count}")
    print(f"  Videos skipped:   {skip_count} (no face detected or unreadable)")
    print(f"  Total frames:     {total_frames}")
    print(f"  Output directory:  {output_dir}")
    print(f"{'=' * 60}")

    # Print final directory stats
    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            label_dir = output_dir / split / label
            if label_dir.exists():
                num_videos = sum(1 for d in label_dir.iterdir() if d.is_dir())
                num_frames_total = sum(
                    len(list(d.glob('*.jpg')))
                    for d in label_dir.iterdir() if d.is_dir()
                )
                print(f"  {split}/{label}: {num_videos} videos, {num_frames_total} frames")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract face frames from video datasets for temporal deepfake training'
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Root directory containing real/ and fake/ video subdirs')
    parser.add_argument('--output_dir', type=str, default='./data/video_frames',
                        help='Output directory for extracted face frames (default: ./data/video_frames)')
    parser.add_argument('--frames_per_video', type=int, default=16,
                        help='Number of frames to extract per video (default: 16)')
    parser.add_argument('--face_size', type=int, default=256,
                        help='Output face crop size in pixels (default: 256)')
    parser.add_argument('--margin', type=int, default=40,
                        help='MTCNN margin around detected face (default: 40)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of videos for validation (default: 0.2)')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Optional: max videos per class (for quick experiments)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    main(args)
