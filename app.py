import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from temporal_model import TemporalDeepfakeModel, load_temporal_model

# ============================================================
# 1. Initialize Device (Use GPU if available)
# ============================================================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

# ============================================================
# 2. Initialize MTCNN for Face Detection
# ============================================================
# keep_all=False: only return the single most prominent face tensor
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=device,
    keep_all=False,
)

# Separate MTCNN for video (detect bounding boxes only, faster)
mtcnn_detect = MTCNN(
    select_largest=False,
    post_process=False,
    device=device,
    keep_all=True,
)

# ============================================================
# 3. Initialize InceptionResnetV1 for DeepFake Classification
# ============================================================
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')

resnet = InceptionResnetV1(
    pretrained='vggface2',
    classify=True,
    num_classes=1,
    device=device
)

# Load fine-tuned weights if a checkpoint exists
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Loading fine-tuned checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    resnet.load_state_dict(checkpoint['model_state_dict'])
    val_metrics = checkpoint.get('val_metrics', {})
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Validation AUC:   {val_metrics.get('auc', '?')}")
    print(f"  Validation Acc:   {val_metrics.get('accuracy', '?')}")
else:
    print(f"WARNING: No fine-tuned checkpoint found at '{CHECKPOINT_PATH}'.")
    print("  Using raw VGGFace2 pretrained weights — accuracy will be limited.")
    print("  Run 'python train.py --train_dir ./data/train --val_dir ./data/val' to fine-tune.")

resnet.eval()  # Set model to evaluation mode

# ============================================================
# 3b. Initialize Temporal Model (if checkpoint exists)
# ============================================================
TEMPORAL_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_temporal_model.pth')
temporal_model = None
temporal_seq_len = 16  # default

if os.path.isfile(TEMPORAL_CHECKPOINT_PATH):
    print(f"Loading temporal model checkpoint: {TEMPORAL_CHECKPOINT_PATH}")
    try:
        # Peek at checkpoint to get config
        _ckpt = torch.load(TEMPORAL_CHECKPOINT_PATH, map_location=device, weights_only=True)
        _temporal_head = _ckpt.get('temporal_head', 'transformer')
        temporal_seq_len = _ckpt.get('seq_len', 16)

        temporal_model = load_temporal_model(
            checkpoint_path=TEMPORAL_CHECKPOINT_PATH,
            temporal_head=_temporal_head,
            device=device,
            seq_len=temporal_seq_len,
        )
        temporal_model.to(device)
        print(f"  Temporal model ready (head={_temporal_head}, seq_len={temporal_seq_len})")
    except Exception as e:
        print(f"WARNING: Failed to load temporal model: {e}")
        temporal_model = None
else:
    print(f"No temporal model found at '{TEMPORAL_CHECKPOINT_PATH}'.")
    print("  Run 'python train_video.py' to train the temporal model.")

# ============================================================
# 4. Pre-instantiate GradCAM ONCE (was re-created per call before)
# ============================================================
target_layers = [resnet.block8.branch1[-1].conv]
grad_cam = GradCAM(model=resnet, target_layers=target_layers)

# Max dimension cap for MTCNN input
MAX_INPUT_DIMENSION = 1280


# ============================================================
# Utility: Resize large images to speed up MTCNN
# ============================================================
def cap_image_size(img_pil, max_dim=MAX_INPUT_DIMENSION):
    """Resize image if its longest side exceeds max_dim. Returns (resized_pil, scale_factor)."""
    w, h = img_pil.size
    longest = max(w, h)
    if longest <= max_dim:
        return img_pil, 1.0
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return img_pil.resize((new_w, new_h), Image.LANCZOS), scale


# ============================================================
# Utility: Run classification + GradCAM on a single face tensor
# ============================================================
def classify_face(face_tensor, threshold=0.5, with_cam=True, use_tta=False):
    """
    Given a raw face tensor from MTCNN (shape [3, H, W], range [0,255]),
    returns (label, confidence, grad_cam_visualization or None).

    Args:
        face_tensor: Face tensor from MTCNN.
        threshold:   Decision boundary (default 0.5).
        with_cam:    Whether to generate GradCAM visualization.
        use_tta:     Test-time augmentation — average original + horizontally flipped.

    NOTE: GradCAM requires gradients, so we do NOT use inference_mode here
    when with_cam=True.
    """
    # Interpolate to 256x256
    face = F.interpolate(
        face_tensor.unsqueeze(0), size=(256, 256),
        mode='bilinear', align_corners=False
    )
    face_for_viz = face.clone() if with_cam else None

    # Normalize from [0, 255] to [-1, 1] for the pretrained model
    face = (face - 127.5) / 128.0
    face = face.to(device)

    visualization = None

    if with_cam:
        # GradCAM requires gradients — cannot use inference_mode
        face_grad = face.detach().requires_grad_(True)
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = grad_cam(input_tensor=face_grad, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Prepare face image for overlay
        face_image_to_plot = face_for_viz.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        face_image_to_plot = np.clip(face_image_to_plot / 255.0, 0, 1).astype(np.float32)
        visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)

    # Model Prediction (no grad needed here)
    with torch.no_grad():
        prediction = resnet(face)
        pred_value = torch.sigmoid(prediction).item()

        # Test-Time Augmentation: average with horizontally flipped version
        if use_tta:
            face_flipped = torch.flip(face, dims=[-1])  # flip width
            prediction_flipped = resnet(face_flipped)
            pred_value_flipped = torch.sigmoid(prediction_flipped).item()
            pred_value = (pred_value + pred_value_flipped) / 2.0

    if pred_value < threshold:
        label = "Real"
        confidence = (1 - pred_value) * 100
    else:
        label = "Fake"
        confidence = pred_value * 100

    return label, confidence, visualization


# ============================================================
# 5. Image Prediction
# ============================================================
def predict_image(input_image, threshold=0.5, use_tta=True):
    """Single-image deepfake detection with GradCAM overlay."""
    if input_image is None:
        return "Error: No image provided.", None

    img_pil = Image.fromarray(input_image)
    img_pil, _ = cap_image_size(img_pil)

    # Detect and extract the face tensor (keep_all=False → single tensor)
    face = mtcnn(img_pil)

    if face is None:
        return "Error: No face detected in the image.", None

    label, confidence, visualization = classify_face(
        face, threshold=threshold, with_cam=True, use_tta=use_tta
    )

    checkpoint_status = "✓ Fine-tuned" if os.path.isfile(CHECKPOINT_PATH) else "⚠ Pretrained only"
    tta_status = "ON" if use_tta else "OFF"

    result_text = (
        f"Prediction: {label} (Confidence: {confidence:.2f}%)\n"
        f"Threshold: {threshold:.2f} | TTA: {tta_status} | Model: {checkpoint_status}"
    )
    return result_text, visualization


# ============================================================
# 6. Video Prediction
# ============================================================
def predict_video(input_video, frame_skip=5, threshold=0.5, use_tta=False):
    """
    Process a video for deepfake detection.

    Args:
        input_video: Path to the uploaded video file.
        frame_skip: Process every N-th frame (higher = faster but less thorough).
        threshold: Decision boundary for real/fake classification.
        use_tta: Whether to use test-time augmentation (slower but more accurate).

    Returns:
        (summary_text, annotated_video_path)
    """
    if input_video is None:
        return "Error: No video provided.", None

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return "Error: Could not open video file.", None

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_skip = max(1, int(frame_skip))

    # Output video — use temp directory
    output_path = os.path.join(tempfile.mkdtemp(), "deepfake_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Statistics
    total_analyzed = 0
    fake_count = 0
    real_count = 0
    no_face_count = 0
    confidence_scores = []

    # Carry forward label for non-analyzed frames
    last_label = None
    last_confidence = 0.0
    last_boxes = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # --- Full analysis on this frame ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_pil_resized, scale = cap_image_size(img_pil)

            # Detect bounding boxes
            boxes, _ = mtcnn_detect.detect(img_pil_resized)

            if boxes is not None and len(boxes) > 0:
                # Extract face tensor for classification (single face)
                face = mtcnn(img_pil_resized)

                if face is not None:
                    label, confidence, _ = classify_face(
                        face, threshold=threshold, with_cam=False, use_tta=use_tta
                    )

                    last_label = label
                    last_confidence = confidence
                    last_boxes = boxes
                    total_analyzed += 1
                    confidence_scores.append(confidence)

                    if label == "Fake":
                        fake_count += 1
                    else:
                        real_count += 1
                else:
                    no_face_count += 1
                    last_boxes = None
            else:
                no_face_count += 1
                last_boxes = None

            # Draw boxes for analyzed frame
            if last_boxes is not None and last_label is not None:
                for box in last_boxes:
                    x1, y1, x2, y2 = [int(coord / scale) for coord in box]
                    # Clamp to frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    color = (0, 0, 255) if last_label == "Fake" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{last_label} {last_confidence:.1f}%"
                    cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Non-analyzed frame: re-use last label, optionally re-detect boxes
            if last_label is not None and last_boxes is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil_resized, scale = cap_image_size(img_pil)
                boxes, _ = mtcnn_detect.detect(img_pil_resized)

                draw_boxes = boxes if (boxes is not None and len(boxes) > 0) else last_boxes

                for box in draw_boxes:
                    x1, y1, x2, y2 = [int(coord / scale) for coord in box]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    color = (0, 0, 255) if last_label == "Fake" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{last_label} {last_confidence:.1f}%"
                    cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # --- Build summary ---
    checkpoint_status = "Fine-tuned" if os.path.isfile(CHECKPOINT_PATH) else "Pretrained only"
    tta_status = "ON" if use_tta else "OFF"

    if total_analyzed == 0:
        summary = (
            f"Processed {frame_idx} frames (analyzed every {frame_skip}th frame).\n"
            f"No faces were detected in any analyzed frame."
        )
    else:
        fake_pct = (fake_count / total_analyzed) * 100
        real_pct = (real_count / total_analyzed) * 100
        avg_confidence = np.mean(confidence_scores)
        verdict = "LIKELY FAKE" if fake_pct > 50 else "LIKELY REAL"

        summary = (
            f"Video Analysis Complete\n"
            f"{'=' * 40}\n"
            f"Total frames: {frame_idx}\n"
            f"Frames analyzed: {total_analyzed} (every {frame_skip}th frame)\n"
            f"Frames with no face: {no_face_count}\n"
            f"{'=' * 40}\n"
            f"Real frames: {real_count} ({real_pct:.1f}%)\n"
            f"Fake frames: {fake_count} ({fake_pct:.1f}%)\n"
            f"Average confidence: {avg_confidence:.1f}%\n"
            f"{'=' * 40}\n"
            f"Overall Verdict: {verdict}\n"
            f"{'=' * 40}\n"
            f"Model: {checkpoint_status} | Threshold: {threshold} | TTA: {tta_status}"
        )

    return summary, output_path


# ============================================================
# 6b. Temporal Video Prediction
# ============================================================
def predict_video_temporal(input_video, threshold=0.5, seq_len=16):
    """
    Process a video using the temporal model for deepfake detection.

    Extracts a sequence of face frames, passes them through the
    InceptionResnetV1 + Transformer/LSTM temporal model for a
    single video-level prediction.

    Args:
        input_video: Path to the uploaded video file.
        threshold:   Decision boundary for real/fake classification.
        seq_len:     Number of frames to sample from the video.

    Returns:
        (summary_text, annotated_video_path)
    """
    if input_video is None:
        return "Error: No video provided.", None

    if temporal_model is None:
        return (
            "Error: Temporal model not available.\n"
            "Train it first: python train_video.py --train_dir ./data/video_frames/train "
            "--val_dir ./data/video_frames/val"
        ), None

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        return "Error: Could not open video file.", None

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        return "Error: Could not read video frames.", None

    # Sample frame indices evenly across the video
    if total_frames <= seq_len:
        frame_indices = list(range(total_frames))
    else:
        start = max(1, int(total_frames * 0.02))
        end = min(total_frames - 1, int(total_frames * 0.98))
        if end <= start:
            start, end = 0, total_frames - 1
        frame_indices = np.linspace(start, end, num=seq_len, dtype=int).tolist()

    # Extract face frames
    face_frames = []
    face_boxes_per_frame = {}  # For annotation
    original_frames = {}  # Store original frames for annotation

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil_resized, scale = cap_image_size(img_pil)

        # Detect face
        face = mtcnn(img_pil_resized)
        if face is not None:
            # Resize face to 256x256
            face_resized = F.interpolate(
                face.unsqueeze(0), size=(256, 256),
                mode='bilinear', align_corners=False
            ).squeeze(0)

            # Normalize from [0, 255] to [-1, 1]
            face_normalized = (face_resized - 127.5) / 128.0
            face_frames.append(face_normalized)

            # Detect bounding boxes for annotation
            boxes, _ = mtcnn_detect.detect(img_pil_resized)
            if boxes is not None:
                face_boxes_per_frame[idx] = (boxes, scale)
            original_frames[idx] = frame.copy()

    cap.release()

    if len(face_frames) == 0:
        return "Error: No faces detected in any sampled frame.", None

    # Pad or truncate to seq_len
    while len(face_frames) < seq_len:
        face_frames.append(face_frames[len(face_frames) % len(face_frames)])
    face_frames = face_frames[:seq_len]

    # Stack into batch: (1, seq_len, 3, 256, 256)
    frames_tensor = torch.stack(face_frames, dim=0).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = temporal_model(frames_tensor).squeeze()
        pred_value = torch.sigmoid(logits).item()

    if pred_value < threshold:
        label = "Real"
        confidence = (1 - pred_value) * 100
    else:
        label = "Fake"
        confidence = pred_value * 100

    # Create annotated output video
    cap = cv2.VideoCapture(input_video)
    output_path = os.path.join(tempfile.mkdtemp(), "temporal_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
    text = f"Temporal: {label} {confidence:.1f}%"

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw label on every frame
        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw face boxes if available for nearby analyzed frames
        nearest_analyzed = min(face_boxes_per_frame.keys(), key=lambda x: abs(x - frame_idx),
                               default=None) if face_boxes_per_frame else None
        if nearest_analyzed is not None and abs(nearest_analyzed - frame_idx) < fps:
            boxes, scale = face_boxes_per_frame[nearest_analyzed]
            for box in boxes:
                x1, y1, x2, y2 = [int(coord / scale) for coord in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Summary
    temporal_status = "Transformer" if temporal_model is not None else "N/A"
    checkpoint_info = os.path.isfile(TEMPORAL_CHECKPOINT_PATH)

    summary = (
        f"Temporal Video Analysis Complete\n"
        f"{'=' * 40}\n"
        f"Model: Temporal ({temporal_status})\n"
        f"Frames sampled: {len(frame_indices)} (of {total_frames} total)\n"
        f"Faces detected: {sum(1 for _ in face_boxes_per_frame)}\n"
        f"{'=' * 40}\n"
        f"Prediction: {label}\n"
        f"Confidence: {confidence:.1f}%\n"
        f"{'=' * 40}\n"
        f"Threshold: {threshold} | Sequence length: {seq_len}"
    )

    return summary, output_path


# ============================================================
# 6c. Video Prediction Router
# ============================================================
def predict_video_router(input_video, frame_skip=5, threshold=0.5, use_tta=False, use_temporal=False):
    """
    Routes video prediction to either frame-by-frame or temporal model.
    """
    if use_temporal:
        return predict_video_temporal(input_video, threshold=threshold, seq_len=temporal_seq_len)
    else:
        return predict_video(input_video, frame_skip=frame_skip, threshold=threshold, use_tta=use_tta)


# ============================================================
# 7. Build the Gradio Interface with Tabs
# ============================================================
with gr.Blocks(
    title="DeepFake Detection Engine",
) as app:
    gr.Markdown(
        """
        # DeepFake Detection Engine
        Upload an **image** or **video** containing faces. The system uses **MTCNN** for face extraction
        and **InceptionResnetV1** for classification, with **Grad-CAM** heatmaps for explainability.
        """
    )

    with gr.Tabs():
        # ---- Image Tab ----
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="Upload Image")
                    img_threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Decision Threshold",
                        info="Lower = more sensitive to fakes, Higher = more conservative"
                    )
                    img_tta = gr.Checkbox(
                        value=True,
                        label="Test-Time Augmentation (TTA)",
                        info="Average prediction with flipped image for better accuracy"
                    )
                    img_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column():
                    img_result = gr.Textbox(label="Detection Result", lines=3)
                    img_cam = gr.Image(label="Grad-CAM Explainability Mask")

            img_btn.click(
                fn=predict_image,
                inputs=[img_input, img_threshold, img_tta],
                outputs=[img_result, img_cam],
            )

        # ---- Video Tab ----
        with gr.TabItem("Video Detection"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Video", format="mp4")
                    vid_temporal = gr.Checkbox(
                        value=temporal_model is not None,
                        label="Use Temporal Model (sequence analysis)",
                        info=(
                            "Analyzes temporal patterns across frames using Transformer. "
                            + ("✓ Model loaded" if temporal_model is not None else "⚠ Not available — train with train_video.py")
                        ),
                        interactive=temporal_model is not None,
                    )
                    vid_skip = gr.Slider(
                        minimum=1, maximum=30, value=5, step=1,
                        label="Frame Skip (analyze every N-th frame)",
                        info="Higher = faster processing, lower = more thorough",
                        visible=temporal_model is None,
                    )
                    vid_threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Decision Threshold",
                        info="Lower = more sensitive to fakes, Higher = more conservative"
                    )
                    vid_tta = gr.Checkbox(
                        value=False,
                        label="Test-Time Augmentation (TTA)",
                        info="More accurate but slower (~2x per frame)",
                        visible=temporal_model is None,
                    )
                    vid_btn = gr.Button("Analyze Video", variant="primary")
                with gr.Column():
                    vid_result = gr.Textbox(label="Analysis Summary", lines=14)
                    vid_output = gr.Video(label="Annotated Output Video", format="mp4")

            # Toggle visibility of frame-by-frame controls based on temporal checkbox
            def toggle_temporal(use_temporal):
                return (
                    gr.update(visible=not use_temporal),  # vid_skip
                    gr.update(visible=not use_temporal),  # vid_tta
                )

            vid_temporal.change(
                fn=toggle_temporal,
                inputs=[vid_temporal],
                outputs=[vid_skip, vid_tta],
            )

            vid_btn.click(
                fn=predict_video_router,
                inputs=[vid_input, vid_skip, vid_threshold, vid_tta, vid_temporal],
                outputs=[vid_result, vid_output],
            )

# ============================================================
# 8. Launch the App
# ============================================================
if __name__ == "__main__":
    app.launch()