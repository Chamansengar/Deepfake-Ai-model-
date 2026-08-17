# 🧠 DeepFake Detection Engine

A deep learning-based deepfake detection system that analyzes both **images** and **videos** using a hybrid spatial-temporal architecture. It combines **MTCNN** face detection, **InceptionResnetV1** spatial classification, and a **Transformer/LSTM** temporal model — all wrapped in an interactive **Gradio** web interface with **Grad-CAM** explainability.

---

## ✨ Features

- 🖼️ **Image Detection** — Analyze a single photo for deepfake manipulation
- 🎬 **Video Detection** — Frame-by-frame analysis with bounding box annotations
- 🕐 **Temporal Video Analysis** — Sequence-aware detection using Transformer or LSTM across multiple frames
- 🔥 **Grad-CAM Explainability** — Heatmap overlays showing what the model focuses on
- 🔄 **Test-Time Augmentation (TTA)** — Optional horizontal flip averaging for improved accuracy
- ⚙️ **Adjustable Thresholds** — Fine-tune sensitivity for real/fake classification
- 🚀 **GPU Acceleration** — Automatic CUDA support when available

---

## 🏗️ Architecture

```
Image/Video
    └─► MTCNN (Face Detection)
            └─► InceptionResnetV1 (Spatial Backbone, VGGFace2 pretrained)
                    ├─► Binary Classifier         → Image prediction + Grad-CAM
                    └─► Frame Embeddings (512-d)
                              └─► Temporal Head
                                    ├─► Transformer Encoder (CLS token)
                                    └─► Bidirectional LSTM
                                              └─► Video-level prediction
```

**Temporal Model pipeline:**
```
Video clip (B, T, 3, 256, 256)
    → InceptionResnetV1 per frame → (B, T, 512) embeddings
    → Temporal Head (Transformer / LSTM)
    → Binary classification (Real / Fake)
```

---

## 📁 Project Structure

```
Ai model/
├── app.py                      # Main Gradio web app
├── temporal_model.py           # TemporalDeepfakeModel (Transformer + LSTM heads)
├── train.py                    # Phase 1: Fine-tune spatial image classifier
├── train_video.py              # Phase 2: Train frame-based temporal model
├── train_video_clips.py        # Phase 2 (alt): Train on raw video clips
├── dataset.py                  # Image dataset & dataloaders
├── video_dataset.py            # Frame-based video dataset
├── video_clip_dataset.py       # Video clip dataset
├── preprocess_videos.py        # Extract face frames from video datasets
├── run_training_pipeline.ps1   # One-shot: runs Phase 1 + Phase 2 training
├── run_video_clips_pipeline.ps1# One-shot: runs clips-based training pipeline
├── requirements.txt            # Python dependencies
├── checkpoints/                # Saved model weights
│   ├── best_model.pth              # Spatial image model checkpoint
│   ├── best_temporal_model.pth     # Frame-based temporal model checkpoint
│   └── best_temporal_clips_model.pth # Video clips temporal model checkpoint
└── data/                       # Training data (not included)
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    ├── videos/                 # Raw video dataset (input for preprocessing)
    │   ├── real/
    │   └── fake/
    └── video_frames/           # Preprocessed frame sequences (output)
        ├── train/
        └── val/
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
| **Video Detection** | Upload a video → analyzed frame-by-frame or via temporal model → annotated video output |

---

## 🏋️ Training Your Own Model

### Phase 1 — Spatial Image Classifier

Fine-tunes **InceptionResnetV1** on your image dataset.

**Data structure required:**
```
data/
├── train/
│   ├── real/   ← real face images
│   └── fake/   ← deepfake images
└── val/
    ├── real/
    └── fake/
```

**Run training:**
```bash
python train.py \
    --train_dir ./data/train \
    --val_dir ./data/val \
    --epochs 20 \
    --batch_size 16 \
    --lr 0.0001
```

Saves the best checkpoint to `checkpoints/best_model.pth` (tracked by validation AUC).

---

### Phase 2 — Temporal Video Model

Requires face frame sequences extracted from videos.

**Step 1: Preprocess videos**
```bash
python preprocess_videos.py \
    --input_dir ./data/videos \
    --output_dir ./data/video_frames \
    --frames_per_video 16 \
    --val_split 0.2
```

**Step 2: Train temporal model**
```bash
# Frame-based pipeline (Transformer head)
python train_video.py \
    --train_dir ./data/video_frames/train \
    --val_dir ./data/video_frames/val \
    --epochs 20 \
    --seq_len 16 \
    --temporal_head transformer \
    --batch_size 4

# Or use LSTM head
python train_video.py \
    --temporal_head lstm
```

---

### 🔁 One-Shot Training Pipeline (PowerShell)

Runs Phase 1 and Phase 2 sequentially:

```powershell
.\run_training_pipeline.ps1
```

Or using the video clips pipeline:

```powershell
.\run_video_clips_pipeline.ps1
```

---

## ⚙️ Configuration & Options

### Image Detection Options

| Option | Default | Description |
|--------|---------|-------------|
| Decision Threshold | `0.5` | Lower → more sensitive to fakes |
| Test-Time Augmentation | `ON` | Averages prediction with horizontally flipped image |

### Video Detection Options

| Option | Default | Description |
|--------|---------|-------------|
| Frame Skip | `5` | Analyze every N-th frame (higher = faster) |
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

## 🧪 Model Performance

After training, the app reports:
- **Validation AUC** — Area Under ROC Curve (higher is better)
- **Validation Accuracy** — % correctly classified
- **Per-frame confidence** — probability score for each analyzed frame
- **Overall Verdict** — `LIKELY REAL` or `LIKELY FAKE` based on majority of analyzed frames

---

## 📊 Data Sources

This model is designed to work with common deepfake benchmark datasets such as:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)

Organize your downloaded dataset into the `data/` folder structure described above.

---

## 🔍 How It Works

1. **Face Extraction** — MTCNN detects and crops the most prominent face from the input
2. **Spatial Classification** — InceptionResnetV1 (fine-tuned on deepfake data) classifies the face
3. **Grad-CAM** — Gradient-based heatmap highlights which facial regions influenced the decision
4. **Temporal Analysis** (video) — A sequence of face frames is passed through a Transformer/LSTM to capture manipulation artifacts that evolve across time
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
