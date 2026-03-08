"""Deep diagnostic: synthetic CTC test + single-sample real data test."""
import os
import json
import argparse
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    read_right_hand_sequence, normalize_frames, normalize_landmarks,
    count_valid_frames, _get_right_hand_cols,
)
from src.data.vocab import build_ctc_vocab, encode_phrase
from src.models.embedded_rnn import EmbeddedRNN
from src.utils.metrics import ctc_greedy_decode


def test_synthetic_ctc():
    """Test 1: Can the model + CTC learn a trivial synthetic task?"""
    print("=" * 60)
    print("TEST 1: SYNTHETIC CTC (should converge to ~0 loss)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = 10  # small vocab: blank(0) + 9 chars
    D = 16  # small input dim
    T = 30  # short sequences
    B = 4

    model = nn.Sequential(
        nn.LSTM(D, 64, batch_first=True, num_layers=1),
    )
    # Custom simple model for test
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(D, 64, batch_first=True, num_layers=1)
            self.fc = nn.Linear(64, C)
        def forward(self, x):
            out, _ = self.rnn(x)
            out = self.fc(out)
            out = torch.nn.functional.log_softmax(out, dim=2)
            return out.permute(1, 0, 2)  # (T, B, C)

    model = SimpleModel().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed synthetic data — same batch every step (overfit test)
    torch.manual_seed(42)
    X = torch.randn(B, T, D).to(device)
    # Targets: each sample has 3-4 characters
    targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3])
    target_lens = torch.tensor([3, 3, 3, 3])
    input_lens = torch.tensor([T, T, T, T])

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        log_probs = model(X)
        loss = criterion(log_probs, targets, input_lens, target_lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}")

    print(f"  Final loss: {loss.item():.4f}")
    if loss.item() < 0.1:
        print("  PASS: Synthetic CTC converges fine.")
    else:
        print("  FAIL: CTC can't even learn synthetic data!")
    print()


def test_embedded_rnn_synthetic():
    """Test 2: Same test but with the actual EmbeddedRNN model."""
    print("=" * 60)
    print("TEST 2: EmbeddedRNN on synthetic data")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C = 28
    D = 63
    T = 60
    B = 4

    model = EmbeddedRNN(D, 256, C).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.manual_seed(42)
    X = torch.randn(B, T, D).to(device)
    targets = torch.tensor([1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20])
    target_lens = torch.tensor([5, 5, 5, 5])
    input_lens = torch.tensor([T, T, T, T])

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        log_probs = model(X)
        loss = criterion(log_probs, targets, input_lens, target_lens)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Epoch {epoch}: NaN/Inf loss!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % 30 == 0:
            # Decode predictions
            model.eval()
            with torch.no_grad():
                preds = model(X)
                pred_ids = torch.argmax(preds[:, 0, :], dim=-1).cpu().tolist()
                blank_ratio = sum(1 for p in pred_ids[:T] if p == 0) / T
            model.train()
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, blank_ratio={blank_ratio:.2f}")

    print(f"  Final loss: {loss.item():.4f}")
    if loss.item() < 1.0:
        print("  PASS: EmbeddedRNN can learn synthetic CTC.")
    else:
        print("  FAIL: EmbeddedRNN can't learn even synthetic CTC!")
    print()


def test_real_single_sample(data_dir):
    """Test 3: Overfit on a single real sample. Print predictions."""
    print("=" * 60)
    print("TEST 3: OVERFIT 1 REAL SAMPLE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_json = os.path.join(data_dir, "character_to_prediction_index.json")
    train_csv = os.path.join(data_dir, "train.csv")
    landmarks_dir = os.path.join(data_dir, "train_landmarks")

    letter_to_int, int_to_letter, blank_id = build_ctc_vocab(vocab_json)
    output_dim = max(int_to_letter.keys()) + 1

    # Find a good sample (with many valid frames)
    df = pd.read_csv(train_csv)
    available = set()
    for fn in os.listdir(landmarks_dir):
        if fn.endswith(".parquet"):
            try:
                available.add(int(fn.replace(".parquet", "")))
            except ValueError:
                pass
    df = df[df["file_id"].isin(available)].copy()
    df["phrase"] = df["phrase"].astype(str).str.lower().str.strip()
    df = df[df["phrase"].apply(lambda x: bool(re.match(r'^[a-z ]+$', x)) and len(x) > 0)].copy()

    # Find sample with high valid frame ratio and short phrase
    best_row = None
    best_ratio = 0
    for _, row in df.head(200).iterrows():
        file_id = int(row["file_id"])
        seq_id = int(row["sequence_id"])
        phrase = row["phrase"]
        if len(phrase) > 10:
            continue
        parquet_path = os.path.join(landmarks_dir, f"{file_id}.parquet")
        X_raw = read_right_hand_sequence(parquet_path, seq_id)
        n_valid = count_valid_frames(X_raw)
        ratio = n_valid / len(X_raw) if len(X_raw) > 0 else 0
        if ratio > best_ratio and n_valid > 0:
            best_ratio = ratio
            best_row = row
            best_X_raw = X_raw
            best_n_valid = n_valid

    if best_row is None:
        print("  No valid sample found!")
        return

    phrase = best_row["phrase"]
    encoded = encode_phrase(phrase, letter_to_int)
    print(f"  Selected sample: phrase='{phrase}', encoded={encoded}")
    print(f"  X_raw shape: {best_X_raw.shape}, valid_frames: {best_n_valid}/{len(best_X_raw)}")

    # Prepare data
    valid_mask = ~np.all(np.isnan(best_X_raw), axis=1)
    X_clean = best_X_raw[valid_mask]
    input_len = min(len(X_clean), 160)
    X = normalize_frames(X_clean, 160)
    X = np.nan_to_num(X, nan=0.0)
    X_norm = normalize_landmarks(X)

    print(f"  After processing: shape={X_norm.shape}, range=[{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print(f"  input_len={input_len}, target_len={len(encoded)}")

    X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 160, 63)
    Y = torch.tensor(encoded, dtype=torch.long)
    in_lens = torch.tensor([input_len])
    tar_lens = torch.tensor([len(encoded)])

    # Also test WITHOUT normalize_landmarks
    X_raw_proc = normalize_frames(X_clean, 160)
    X_raw_proc = np.nan_to_num(X_raw_proc, nan=0.0)
    X_raw_t = torch.tensor(X_raw_proc, dtype=torch.float32).unsqueeze(0).to(device)

    print(f"\n  --- Training WITH normalize_landmarks ---")
    _train_single(X_t, Y, in_lens, tar_lens, int_to_letter, blank_id, output_dim, device, "normalized")

    print(f"\n  --- Training WITHOUT normalize_landmarks ---")
    _train_single(X_raw_t, Y, in_lens, tar_lens, int_to_letter, blank_id, output_dim, device, "raw")


def _train_single(X, Y, in_lens, tar_lens, int_to_letter, blank_id, output_dim, device, label):
    model = EmbeddedRNN(63, 256, output_dim).to(device)
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    target_text = "".join(int_to_letter[int(t)] for t in Y.tolist())

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        log_probs = model(X)  # (T, 1, C)
        loss = criterion(log_probs, Y, in_lens, tar_lens)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Epoch {epoch}: NaN/Inf!")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % 30 == 0 or epoch == 299:
            model.eval()
            with torch.no_grad():
                preds = model(X)
                valid_t = int(in_lens[0].item())
                pred_text = ctc_greedy_decode(preds[:valid_t, 0, :], int_to_letter, blank_id)
                pred_ids = torch.argmax(preds[:valid_t, 0, :], dim=-1).cpu().tolist()
                blank_ratio = sum(1 for p in pred_ids if p == blank_id) / len(pred_ids)
            print(f"  [{label}] Epoch {epoch:3d}: loss={loss.item():.4f}, blank_ratio={blank_ratio:.2f}")
            print(f"    GT:   '{target_text}'")
            print(f"    PRED: '{pred_text}'")

    print(f"  [{label}] Final loss: {loss.item():.4f}")


def check_parquet_columns(data_dir):
    """Check what columns the parquet files actually have."""
    print("=" * 60)
    print("TEST 0: PARQUET COLUMN CHECK")
    print("=" * 60)
    landmarks_dir = os.path.join(data_dir, "train_landmarks")
    parquet_files = [f for f in os.listdir(landmarks_dir) if f.endswith(".parquet")]
    if not parquet_files:
        print("  No parquet files found!")
        return

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(os.path.join(landmarks_dir, parquet_files[0]))
    all_cols = pf.schema.names
    right_hand_cols = [c for c in all_cols if "right_hand" in c]

    print(f"  Total columns: {len(all_cols)}")
    print(f"  Right hand columns: {len(right_hand_cols)}")
    print(f"  First 10 right_hand cols: {right_hand_cols[:10]}")
    print(f"  Left hand columns: {len([c for c in all_cols if 'left_hand' in c])}")
    print(f"  Face columns: {len([c for c in all_cols if 'face' in c])}")
    print(f"  Pose columns: {len([c for c in all_cols if 'pose' in c])}")

    # Check column naming pattern
    if right_hand_cols:
        print(f"\n  Column naming pattern:")
        for c in right_hand_cols[:6]:
            print(f"    {c}")
        print(f"    ...")
        for c in right_hand_cols[-3:]:
            print(f"    {c}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    args = p.parse_args()

    check_parquet_columns(args.data_dir)
    print()
    test_synthetic_ctc()
    test_embedded_rnn_synthetic()
    test_real_single_sample(args.data_dir)


if __name__ == "__main__":
    main()
