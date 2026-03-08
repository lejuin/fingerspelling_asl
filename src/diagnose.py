"""Diagnostic script to find the root cause of CTC blank collapse."""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import (
    ASLRightHandDataset, collate_fn, read_right_hand_sequence,
    normalize_frames, normalize_landmarks, count_valid_frames,
)
from src.data.vocab import build_ctc_vocab, encode_phrase
from src.models.embedded_rnn import EmbeddedRNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    args = p.parse_args()

    vocab_json = os.path.join(args.data_dir, "character_to_prediction_index.json")
    train_csv = os.path.join(args.data_dir, "train.csv")
    landmarks_dir = os.path.join(args.data_dir, "train_landmarks")

    # 1. Check vocab
    print("=" * 60)
    print("1. VOCAB CHECK")
    print("=" * 60)
    with open(vocab_json) as f:
        raw_vocab = json.load(f)
    print(f"Raw vocab size: {len(raw_vocab)}")
    print(f"First 5 entries: {dict(list(raw_vocab.items())[:5])}")

    letter_to_int, int_to_letter, blank_id = build_ctc_vocab(vocab_json)
    print(f"CTC vocab size (with blank): {len(int_to_letter) + 1}")
    print(f"blank_id={blank_id}")
    print(f"letter_to_int: {letter_to_int}")
    print(f"int_to_letter: {int_to_letter}")
    output_dim = max(int_to_letter.keys()) + 1
    print(f"output_dim={output_dim}")

    # 2. Check a few raw samples
    print("\n" + "=" * 60)
    print("2. RAW DATA CHECK")
    print("=" * 60)
    df = pd.read_csv(train_csv)
    # Filter to available parquets
    available = set()
    for fn in os.listdir(landmarks_dir):
        if fn.endswith(".parquet"):
            try:
                available.add(int(fn.replace(".parquet", "")))
            except ValueError:
                pass
    df = df[df["file_id"].isin(available)].copy()
    df["phrase"] = df["phrase"].astype(str).str.lower().str.strip()
    import re
    df = df[df["phrase"].apply(lambda x: bool(re.match(r'^[a-z ]+$', x)) and len(x) > 0)].copy()
    print(f"Available samples: {len(df)}")

    # Take 5 samples
    samples = df.head(5)
    for i, (_, row) in enumerate(samples.iterrows()):
        file_id = int(row["file_id"])
        seq_id = int(row["sequence_id"])
        phrase = row["phrase"]
        encoded = encode_phrase(phrase, letter_to_int)

        parquet_path = os.path.join(landmarks_dir, f"{file_id}.parquet")
        X_raw = read_right_hand_sequence(parquet_path, seq_id)

        n_valid = count_valid_frames(X_raw)
        n_total = len(X_raw)
        nan_pct = np.isnan(X_raw).mean() * 100

        print(f"\nSample {i}: file_id={file_id}, seq_id={seq_id}")
        print(f"  Phrase: '{phrase}' (len={len(phrase)})")
        print(f"  Encoded: {encoded} (len={len(encoded)})")
        print(f"  X_raw shape: {X_raw.shape}")
        print(f"  Total frames: {n_total}, Valid frames: {n_valid}, NaN%: {nan_pct:.1f}%")
        print(f"  X_raw value range (ignoring NaN): [{np.nanmin(X_raw):.4f}, {np.nanmax(X_raw):.4f}]")
        print(f"  X_raw mean (ignoring NaN): {np.nanmean(X_raw):.4f}")

        # After processing
        X = normalize_frames(X_raw, 160)
        X = np.nan_to_num(X, nan=0.0)
        print(f"  After nan_to_num - shape: {X.shape}, range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"  After nan_to_num - zero_pct: {(X == 0).mean() * 100:.1f}%")

        X_norm = normalize_landmarks(X)
        print(f"  After normalize_landmarks - range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
        print(f"  After normalize_landmarks - mean: {X_norm.mean():.4f}, std: {X_norm.std():.4f}")
        print(f"  After normalize_landmarks - zero_pct: {(X_norm == 0).mean() * 100:.1f}%")

        # Check a valid frame vs padded frame
        if n_valid > 0 and n_total < 160:
            print(f"  Frame 0 (should be valid): min={X_norm[0].min():.4f}, max={X_norm[0].max():.4f}, nonzero={np.count_nonzero(X_norm[0])}")
            print(f"  Frame {n_total} (should be padding): min={X_norm[n_total].min():.4f}, max={X_norm[n_total].max():.4f}, nonzero={np.count_nonzero(X_norm[n_total])}")

        # CTC constraint check
        input_len = min(n_total, 160)
        target_len = len(encoded)
        print(f"  CTC check: input_len={input_len} >= target_len={target_len}? {input_len >= target_len}")

    # 3. Model forward pass check
    print("\n" + "=" * 60)
    print("3. MODEL FORWARD PASS CHECK")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare a mini dataset
    mini_df = df.head(8).copy()
    mini_df["encoded"] = mini_df["phrase"].apply(lambda x: encode_phrase(str(x), letter_to_int))
    ds = ASLRightHandDataset(mini_df, landmarks_dir=landmarks_dir, max_frames=160, training=False)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = EmbeddedRNN(63, 256, output_dim).to(device)
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    for batch in loader:
        if batch is None:
            print("Batch is None!")
            continue
        X, Y, in_lens, tar_lens = batch
        X = X.to(device)
        print(f"Batch X shape: {X.shape}")
        print(f"Batch X range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"Batch X mean: {X.mean():.4f}, std: {X.std():.4f}")
        print(f"Y: {Y}")
        print(f"in_lens: {in_lens}")
        print(f"tar_lens: {tar_lens}")

        log_probs = model(X)  # (T, B, C)
        print(f"\nModel output shape: {log_probs.shape}")
        print(f"Model output range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")

        # Check probability distribution at first time step
        probs = torch.exp(log_probs[0, 0, :])
        print(f"Probs at t=0, b=0: sum={probs.sum():.4f}")
        print(f"  blank prob: {probs[blank_id]:.4f}")
        print(f"  max non-blank prob: {probs[1:].max():.4f}")
        print(f"  argmax: {torch.argmax(probs).item()}")

        # CTC loss
        loss = criterion(log_probs, Y, in_lens, tar_lens)
        print(f"\nCTC loss: {loss.item():.4f}")
        print(f"ln(output_dim)={np.log(output_dim):.4f} (random baseline)")

        # Check if loss is reasonable
        if loss.item() > 100:
            print("WARNING: Loss is extremely high! Possible input_lengths issue.")
            print("  Checking: all(in_lens >= tar_lens)?", torch.all(in_lens >= tar_lens).item())
            print("  Checking: all(in_lens <= T)?", torch.all(in_lens <= log_probs.shape[0]).item())

        break

    print("\n" + "=" * 60)
    print("4. GRADIENT CHECK (1 step)")
    print("=" * 60)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for batch in loader:
        if batch is None:
            continue
        X, Y, in_lens, tar_lens = batch
        X = X.to(device)

        optimizer.zero_grad()
        log_probs = model(X)
        loss = criterion(log_probs, Y, in_lens, tar_lens)
        print(f"Loss before step: {loss.item():.4f}")
        loss.backward()

        # Check gradients
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                gn = param.grad.norm().item()
                grad_norms.append((name, gn))
                print(f"  {name}: grad_norm={gn:.6f}")

        optimizer.step()

        # Forward again
        with torch.no_grad():
            log_probs2 = model(X)
            loss2 = criterion(log_probs2, Y, in_lens, tar_lens)
            print(f"Loss after step: {loss2.item():.4f}")
            print(f"Loss delta: {loss2.item() - loss.item():.4f}")

        break

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
