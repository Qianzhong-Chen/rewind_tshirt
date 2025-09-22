#!/usr/bin/env python3
# Compute MSE between smoothed.npy and gt.npy across all episodes in a run directory.
# Hardcoded eval dir per request.

import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional

# >>> Hardcoded path <<<
# BASE_DIR = "/home/david_chen/rewind_tshirt/outputs/2025-08-28/07-05-01/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_regression_sparse/eval_video/2025.08.28-07.05.20"
# BASE_DIR = "/home/david_chen/rewind_tshirt/outputs/2025-08-28/07-09-50/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/eval_video/2025.08.28-07.10.12"
# BASE_DIR = "/home/david_chen/rewind_tshirt/outputs/2025-09-03/01-31-18/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/eval_video/2025.09.03-01.31.45"
# BASE_DIR = "/home/david_chen/rewind_tshirt/outputs/2025-09-03/13-44-00/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_gvl_sparse/eval_video/2025.09.03-13.44.20"
# BASE_DIR = "/home/david_chen/rewind_tshirt/outputs/2025-09-04/16-59-37/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_vlc_sparse/eval_video/2025.09.04-17.00.06"
BASE_DIR = "/home/david_chen/rewind_tshirt/outputs/2025-09-05/21-22-22/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_liv_sparse/eval_video/2025.09.05-21.22.48"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_folding_ablation/2025-09-08/12-48-57/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_dense_annotation/eval_video/2025.09.08-12.49.11"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_folding_ablation/2025-09-08/13-34-05/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_dense_annotation/eval_video/2025.09.08-13.34.20"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/2025-09-08/15-55-46/rewind_reward_fixed_seq_model_frame_gap_multi_stage/fold_tshirt_hybird/eval_video/2025.09.08-15.56.12"

BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_ablation/2025-09-20/12-14-43/use_state/eval_video/2025.09.20-12.14.57"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_ablation/2025-09-20/12-19-22/wrist_cam/eval_video/2025.09.20-12.19.42"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_ablation/2025-09-20/12-22-44/n_obs_4/eval_video/2025.09.20-12.23.20"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_ablation/2025-09-20/12-25-49/frame_gap_15/eval_video/2025.09.20-12.26.25"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_ablation/2025-09-20/12-29-10/n_obs_12/eval_video/2025.09.20-12.29.58"
BASE_DIR = "/nfs_us/david_chen/reward_model_ckpt/tshirt_ablation/2025-09-20/12-43-45/frame_gap_60/eval_video/2025.09.20-12.44.31" \
""
def load_pair(ep_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    sm_path = ep_dir / "smoothed.npy"
    # sm_path = ep_dir / "pred.npy"
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
