# 🧠 DeepFake Detection Engine

A deep learning-based deepfake detection system that analyses **videos** using a hybrid spatial-temporal architecture. It combines **MTCNN** face detection, **InceptionResnetV1** spatial embeddings, and a **Transformer/LSTM** temporal model — all wrapped in an interactive **Gradio** web interface with **Grad-CAM** explainability.

---

## ✨ Features

- 🖼️ **Image Detection** — Analyse a single photo for deepfake manipulation
- 🎬 **Video Detection** — Frame-by-frame analysis with bounding box annotations
- 🕐 **Temporal Video Analysis** — Sequence-aware detection using Transformer or LSTM across multiple frames
- 🔥 **Grad-CAM Explainability** — Heatmap overlays showing what the model focuses on
- 🔄 **Test-Time Augmentation (TTA)** — Optional horizontal flip averaging for improved accuracy
- ⚙️ **Adjustable Thresholds** — Fine-tune sensitivity for real/fake classification
- 🚀 **GPU Acceleration** — Automatic CUDA + AMP support when available
- ⚡ **Face Caching** — Extracted faces cached to disk for faster subsequent epochs

---

## 🏗️ Architecture

```
Image/Video
    └─► MTCNN (Face Detection)
            └─► InceptionResnetV1 (Spatial Backbone, VGGFace2 pretrained)
                    └─► Frame Embeddings (512-d)
                              └─► Temporal Head
                                    ├─► Transformer Encoder (CLS token)
                                    └─► Bidirectional LSTM
                                              └─► Video-level prediction (Real / Fake)
```

**Temporal Model pipeline:**
```
Raw video (.mp4 / .avi)
    → MTCNN face extraction per frame
    → InceptionResnetV1 per frame → (B, T, 512) embeddings
    → Temporal Head (Transformer / LSTM)
    → Binary classification (Real / Fake)
```

---

## 📁 Project Structure

```
Ai model/
├── app.py                       # Main Gradio web app
├── temporal_model.py            # TemporalDeepfakeModel (Transformer + LSTM heads)
├── train_video_clips.py         # Training script — end-to-end from raw video clips
├── dataset.py                   # Augmentation transforms (reused by clip dataset)
├── video_clip_dataset.py        # VideoClipDataset — reads .mp4/.avi at runtime
├── run_video_clips_pipeline.ps1 # One-shot PowerShell launcher
├── requirements.txt             # Python dependencies
├── checkpoints/                 # Saved model weights
│   └── best_temporal_clips_model.pth  # Best video-clips temporal model
└── data/                        # Training data (not included)
    └── videos/
        ├── real/   ← real face videos (.mp4, .avi, ...)
        └── fake/   ← deepfake videos
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- (Optional) NVIDIA GPU with CUDA for faster processing

### 1. Clone the Repository

```bash
git clone https://github.com/Chamansengar/Deepfake-Ai-model-.git
cd "Ai model"
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the App

```bash
python app.py
```

The Gradio interface will open in your browser. You can:

| Tab | What it does |
|-----|-------------|
| **Image Detection** | Upload a photo → face is extracted → classified as Real/Fake with Grad-CAM heatmap |
| **Video Detection** | Upload a video → analysed frame-by-frame or via temporal model → annotated video output |

---

## 🏋️ Training Your Own Model

### Single-step end-to-end pipeline

`train_video_clips.py` reads raw video files directly — no preprocessing step needed. MTCNN face extraction runs on the fly (with optional disk caching for speed).

**Data structure required:**
```
data/videos/
├── real/   ← real face videos (.mp4, .avi, .mov, ...)
└── fake/   ← deepfake videos
```

**Auto-split mode (80/20 train/val):**
```bash
python train_video_clips.py \
    --video_dir ./data/videos \
    --val_split 0.2 \
    --epochs 30 \
    --seq_len 16 \
    --temporal_head transformer \
    --batch_size 4
```

**Separate train/val directories:**
```bash
python train_video_clips.py \
    --train_video_dir ./data/videos_train \
    --val_video_dir   ./data/videos_val \
    --epochs 30
```

**Enable face caching for faster subsequent epochs:**
```bash
python train_video_clips.py \
    --video_dir ./data/videos \
    --cache_dir ./data/face_cache \
    --epochs 30
```

**Enable mixed-precision training (GPU only):**
```bash
python train_video_clips.py \
    --video_dir ./data/videos \
    --amp \
    --epochs 30
```

**Quick experiment with limited data:**
```bash
python train_video_clips.py \
    --video_dir ./data/videos \
    --max_videos 10 \
    --epochs 3
```

Saves the best checkpoint to `checkpoints/best_temporal_clips_model.pth` (tracked by validation AUC).

---

### 🔁 One-Shot Pipeline (PowerShell)

```powershell
.\run_video_clips_pipeline.ps1
```

---

## 📋 Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | — | Single dir with `real/` and `fake/` (auto-split) |
| `--train_video_dir` / `--val_video_dir` | — | Pre-split directories |
| `--temporal_head` | `transformer` | `transformer` or `lstm` |
| `--seq_len` | `16` | Frames per training sequence |
| `--num_extract_frames` | `32` | Frames extracted from each video |
| `--clips_per_video` | `1` | Number of clips sampled per video per epoch |
| `--epochs` | `30` | Training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `5e-4` | Learning rate |
| `--patience` | `7` | Early stopping patience |
| `--amp` | off | Enable automatic mixed precision (CUDA only) |
| `--cache_dir` | off | Directory to cache extracted faces |
| `--skip_face_detection` | off | Use raw frames instead of MTCNN crops |
| `--unfreeze_backbone` | off | Fine-tune entire backbone end-to-end |
| `--backbone_weights` | auto | Path to fine-tuned backbone checkpoint |

---

## ⚙️ Configuration & App Options

### Image Detection Options

| Option | Default | Description |
|--------|---------|-------------|
| Decision Threshold | `0.5` | Lower → more sensitive to fakes |
| Test-Time Augmentation | `ON` | Averages prediction with horizontally flipped image |

### Video Detection Options

| Option | Default | Description |
|--------|---------|-------------|
| Frame Skip | `5` | Analyse every N-th frame (higher = faster) |
| Decision Threshold | `0.5` | Classification boundary |
| TTA | `OFF` | Test-time augmentation (~2x slower) |
| Temporal Model | Auto | Uses Transformer/LSTM if checkpoint is available |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `facenet-pytorch` | MTCNN face detection + InceptionResnetV1 |
| `grad-cam` | Gradient-weighted Class Activation Maps |
| `gradio` | Interactive web UI |
| `opencv-python` | Video processing |
| `Pillow` | Image handling |
| `numpy` | Numerical computations |
| `scikit-learn` | AUC, accuracy, precision/recall metrics |

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Performance Metrics

After training, the app reports:
- **Validation AUC** — Area Under ROC Curve (higher is better)
- **Validation Accuracy** — % correctly classified
- **Precision / Recall** — Class-specific performance
- **Per-frame confidence** — probability score for each analysed frame
- **Overall Verdict** — `LIKELY REAL` or `LIKELY FAKE` based on majority of analysed frames

---

## 📊 Supported Datasets

This model is designed to work with common deepfake benchmark datasets:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)

Organise your downloaded dataset into the `data/videos/` folder structure described above.

---

## 🔍 How It Works

1. **Face Extraction** — MTCNN detects and crops the most prominent face from each video frame
2. **Spatial Encoding** — InceptionResnetV1 converts each face crop into a 512-d embedding
3. **Temporal Analysis** — A sequence of frame embeddings is passed through a Transformer/LSTM to capture manipulation artefacts that evolve across time
4. **Grad-CAM** — Gradient-based heatmap highlights which facial regions influenced the decision
5. **Verdict** — Final prediction with confidence score and visual annotations

---

## 📝 License

This project is intended for research and educational purposes.

---

## 🙏 Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN & InceptionResnetV1 implementations
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Grad-CAM explainability
- [Gradio](https://gradio.app/) — Web interface framework
- VGGFace2 dataset for pretrained backbone weights
