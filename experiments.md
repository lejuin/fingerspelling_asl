# ASL Fingerspelling — Experiment Tracking

| Run Name | Date | Model | Epochs | Batch Size | LR | Hidden Dim | Proj Dim | RNN Layers | Dropout | Train Size | Val Size | Early Stop | Exec Time (h) | Best val CER | Best val WER | Epoch @ Best | Notes | W&B Link |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clearml-l4-merge | 2026-03-13 | lstm | 45 | 256 | 1e-3 | 512 | 256 | 3 | 0.5 | 50000 | 50000 | no | — | 1.0 | 1.0 | 1 | train_size cap limited real data to ~8k rows; LR too aggressive | — |
| clearml-l4-fulldata | 2026-03-13 | lstm | 45 | 64 | 3e-4 | 512 | 256 | 3 | 0.5 | 0 (all) | 0 (all) | no | — | — | — | — | Full dataset, lower LR; results pending | — |
