#!/usr/bin/env python3
"""Compute mean and std of actions and state from a LeRobot dataset and write statistics.json."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(dataset_path: str, output_path: str, keys=("actions", "state")):
    root = Path(dataset_path)
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}/data/")

    print(f"Found {len(parquet_files)} parquet file(s)")

    frames = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(df)}")

    stats = {}
    for key in keys:
        if key not in df.columns:
            print(f"  [skip] '{key}' not in dataset")
            continue
        # Each cell is a list/array of floats
        arr = np.stack(df[key].tolist())   # [N, dim]
        mean = arr.mean(axis=0).tolist()
        std  = arr.std(axis=0).tolist()
        # Clamp std to avoid division by zero
        std = [max(s, 1e-6) for s in std]
        stats[key] = {"mean": mean, "std": std}
        print(f"  {key}: dim={arr.shape[1]}  mean={[f'{v:.4f}' for v in mean]}  std={[f'{v:.4f}' for v in std]}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="configs/train/statistics.json")
    parser.add_argument("--keys", nargs="+", default=["action", "state"])
    args = parser.parse_args()
    compute_stats(args.dataset_path, args.output, args.keys)
