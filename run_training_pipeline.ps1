$ErrorActionPreference = "Stop"

Write-Host "========================================="
Write-Host " Starting Phase 1: Spatial Image Training"
Write-Host "========================================="
.venv\Scripts\python.exe -u train.py --train_dir ./data/train --val_dir ./data/val --epochs 5 --batch_size 16 --lr 0.0001 --resume ./checkpoints/best_model.pth
if ($LASTEXITCODE -ne 0) { Write-Host "Phase 1 failed!" ; exit 1 }

Write-Host "========================================="
Write-Host " Starting Phase 2: Temporal Video Training"
Write-Host "========================================="
.venv\Scripts\python.exe -u train_video.py --train_dir ./data/video_frames/train --val_dir ./data/video_frames/val --epochs 5 --seq_len 16 --temporal_head transformer --batch_size 4 --lr 0.0005 --resume ./checkpoints/best_temporal_model.pth
if ($LASTEXITCODE -ne 0) { Write-Host "Phase 2 failed!" ; exit 1 }

Write-Host "========================================="
Write-Host " Training Pipeline Complete!"
Write-Host " Checkpoints are saved in ./checkpoints/"
Write-Host "========================================="
