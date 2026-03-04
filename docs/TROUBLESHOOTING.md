# Troubleshooting

## 1) `python` not found
### Symptoms
- `python is not recognized...`

### Fix
1. Install Python 3.10+.
2. Reopen terminal.
3. Check:
```powershell
python --version
```

## 2) Virtual environment not activating (Windows)
### Symptoms
- `running scripts is disabled on this system`

### Fix
Open PowerShell as user and run:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Then:
```powershell
.\.venv\Scripts\activate
```

## 3) Missing dependencies
### Symptoms
- `ModuleNotFoundError: No module named ...`

### Fix
```powershell
.\setup.ps1
```
Or manually:
```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 4) Webcam cannot open
### Symptoms
- `No se pudo abrir la webcam`
- black window

### Fix
1. Close apps using the camera (Teams/Zoom/Meet).
2. Verify OS camera permission for terminal/VS Code.
3. Try external webcam or different camera index in code.

## 5) MediaPipe model file missing
### Symptoms
- `No encuentro el modelo hand_landmarker.task`

### Fix
Put file at:
```text
artifacts/models/hand_landmarker.task
```

## 6) `gcloud compute ssh` permission errors
### Symptoms
- missing `compute.instances.use`
- missing `iam.serviceAccounts.actAs`

### Required IAM roles (project)
- `roles/compute.osAdminLogin` (or `osLogin`)
- `roles/compute.viewer`
- `roles/compute.instanceAdmin.v1`
- `roles/iam.serviceAccountUser`
- `roles/iap.tunnelResourceAccessor` (if using IAP / browser tunnel)

## 7) GPU VM keeps generating cost
### Fix
Stop instance when not used:
```powershell
gcloud compute instances stop instance-fingerspeling --zone=us-central1-b --project=buoyant-purpose-479417-t8
```

## 8) Run appears stuck
### Quick checks
```bash
nvidia-smi
tail -n 80 artifacts/logs/<run_name>.log
```

## 9) W&B not logging
### Symptoms
- run starts without sync

### Fix
```bash
wandb login
python -m src.train --use_wandb --wandb_mode online --wandb_project fingerspelling_asl
```

## 10) Large files accidentally staged for git
### Fix
```powershell
git status
git restore --staged <path>
```
Ensure `.gitignore` includes `data/`, `*.parquet`, `*.pt`, `wandb/`.
