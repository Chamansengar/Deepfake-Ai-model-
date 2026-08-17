$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================"
Write-Host " Video Clips Training Pipeline — Deepfake Detection"
Write-Host " Trains the temporal model directly on raw video files"
Write-Host "================================================================"
Write-Host ""

# ---- Configuration ----
$VIDEO_DIR       = "./data/videos"
$CACHE_DIR       = "./data/face_cache"
$CHECKPOINT_DIR  = "./checkpoints"
$EPOCHS          = 30
$SEQ_LEN         = 16
$BATCH_SIZE      = 4
$LR              = 0.0005
$TEMPORAL_HEAD   = "transformer"
$CLIPS_PER_VIDEO = 1
$EXTRACT_FRAMES  = 32
$VAL_SPLIT       = 0.2

# ---- Optional: Run spatial backbone training first ----
$BACKBONE_CKPT = "$CHECKPOINT_DIR/best_model.pth"
if (-Not (Test-Path $BACKBONE_CKPT)) {
    Write-Host "========================================="
    Write-Host " Phase 0: No backbone checkpoint found."
    Write-Host " Training spatial backbone first..."
    Write-Host "========================================="

    $TRAIN_DIR = "./data/train"
    $VAL_DIR   = "./data/val"

    if ((Test-Path $TRAIN_DIR) -and (Test-Path $VAL_DIR)) {
        .venv\Scripts\python.exe -u train.py `
            --train_dir $TRAIN_DIR `
            --val_dir $VAL_DIR `
            --epochs 5 `
            --batch_size 16 `
            --lr 0.0001
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Phase 0 (backbone training) failed!"
            exit 1
        }
        Write-Host ""
    } else {
        Write-Host "  Skipping — no image training data found at $TRAIN_DIR."
        Write-Host "  The video pipeline will use the raw VGGFace2 backbone."
        Write-Host ""
    }
}

# ---- Phase 1: Train temporal model on raw video clips ----
Write-Host "================================================================"
Write-Host " Phase 1: Training Temporal Model on Raw Video Clips"
Write-Host "================================================================"
Write-Host "  Video directory:   $VIDEO_DIR"
Write-Host "  Face cache:        $CACHE_DIR"
Write-Host "  Temporal head:     $TEMPORAL_HEAD"
Write-Host "  Sequence length:   $SEQ_LEN"
Write-Host "  Batch size:        $BATCH_SIZE"
Write-Host "  Epochs:            $EPOCHS"
Write-Host ""

# Check that video data exists
if (-Not (Test-Path "$VIDEO_DIR/real") -or -Not (Test-Path "$VIDEO_DIR/fake")) {
    Write-Host "ERROR: Expected '$VIDEO_DIR/real/' and '$VIDEO_DIR/fake/' directories."
    Write-Host "  Place your real videos in '$VIDEO_DIR/real/' and fake videos in '$VIDEO_DIR/fake/'."
    exit 1
}

# Resume from existing checkpoint if available
$RESUME_ARG = ""
$CLIPS_CKPT = "$CHECKPOINT_DIR/best_temporal_clips_model.pth"
if (Test-Path $CLIPS_CKPT) {
    $RESUME_ARG = "--resume $CLIPS_CKPT"
    Write-Host "  Resuming from existing checkpoint: $CLIPS_CKPT"
    Write-Host ""
}

.venv\Scripts\python.exe -u train_video_clips.py `
    --video_dir $VIDEO_DIR `
    --val_split $VAL_SPLIT `
    --cache_dir $CACHE_DIR `
    --temporal_head $TEMPORAL_HEAD `
    --seq_len $SEQ_LEN `
    --num_extract_frames $EXTRACT_FRAMES `
    --clips_per_video $CLIPS_PER_VIDEO `
    --batch_size $BATCH_SIZE `
    --lr $LR `
    --epochs $EPOCHS `
    $RESUME_ARG

if ($LASTEXITCODE -ne 0) {
    Write-Host "Video clips training failed!"
    exit 1
}

# ---- Summary ----
Write-Host ""
Write-Host "================================================================"
Write-Host " Video Clips Training Pipeline Complete!"
Write-Host ""
Write-Host " Checkpoints saved in: $CHECKPOINT_DIR/"
Write-Host "   - best_temporal_clips_model.pth  (best by val AUC)"
Write-Host ""
Write-Host " To use in the app:"
Write-Host "   python app.py"
Write-Host "================================================================"
