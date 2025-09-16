#!/usr/bin/env python3
# Compute MSE between smoothed.npy and gt.npy across all episodes in a run directory.
# Hardcoded eval dir per request.

import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional

# >>> Hardcoded path <<<
BASE_DIR = "/nfs_old/david_chen/reward_model_ckpt/dish_unloading/2025-09-16/09-06-22/unload_dish_regression/eval_video"


def load_pair(ep_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # sm_path = ep_dir / "smoothed.npy"
    sm_path = ep_dir / "pred.npy"
    gt_path = ep_dir / "gt.npy"
    if not (sm_path.exists() and gt_path.exists()):
        return None
    try:
        sm = np.load(sm_path, allow_pickle=False)
        gt = np.load(gt_path, allow_pickle=False)
        sm = np.asarray(sm, dtype=np.float64).reshape(-1)
        gt = np.asarray(gt, dtype=np.float64).reshape(-1)
        L = min(len(sm), len(gt))
        if L == 0:
            return None
        sm = sm[:L]
        gt = gt[:L]
        mask = np.isfinite(sm) & np.isfinite(gt)
        if not np.any(mask):
            return None
        return sm[mask], gt[mask]
    except Exception:
        return None


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


def find_episodes(base: Path) -> List[Path]:
    eps = []
    for p in sorted(base.iterdir()):
        # if p.is_dir() and (p / "smoothed.npy").exists() and (p / "gt.npy").exists():
        if p.is_dir() and (p / "gt.npy").exists():
            eps.append(p)
    return eps


def main():
    base = Path(os.path.expanduser(BASE_DIR)).resolve()
    if not base.exists():
        print(f"[ERROR] Path does not exist: {base}")
        raise SystemExit(1)

    episodes = find_episodes(base)
    if not episodes:
        print(f"[WARN] No episodes with smoothed.npy & gt.npy found under: {base}")
        raise SystemExit(0)

    per_ep = []
    total_sse = 0.0
    total_count = 0

    print(f"Scanning {len(episodes)} episode(s) under:\n  {base}\n")
    for ep in episodes:
        pair = load_pair(ep)
        if pair is None:
            print(f"[SKIP] {ep.name}: missing/invalid arrays or no finite overlap.")
            continue
        sm, gt = pair
        ep_mse = mse(sm, gt)
        per_ep.append((ep.name, ep_mse, len(sm)))
        print(f"{ep.name}  MSE = {ep_mse:.6f}")
        total_sse += float(np.sum((sm - gt) ** 2))
        total_count += len(sm)
        # print(f"{ep.name:>16}  MSE = {ep_mse:.6f}  (N={len(sm)})")

    if not per_ep:
        print("\n[WARN] No valid episodes to aggregate.")
        return

    macro_avg = float(np.mean([m for _, m, _ in per_ep]))
    micro_avg = (total_sse / total_count) if total_count > 0 else float("nan")

    print("\nSummary:")
    print(f"  Episodes counted : {len(per_ep)}")
    print(f"  Macro-average MSE (mean of episode MSEs): {macro_avg:.6f}")
    print(f"  Micro-average MSE (global over all samples): {micro_avg:.6f}")


if __name__ == "__main__":
    main()
