#!/usr/bin/env python3
import os
import re
import numpy as np

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Set your eval_video directory path here:
ROOT = "/nfs_us/david_chen/reward_model_ckpt/dish_unloading/2025-09-14/18-53-12/unload_dish/eval_video/2025.09.14-18.53.25"
ROOT = "/nfs_us/david_chen/reward_model_ckpt/dish_unloading/2025-09-14/18-06-57/unload_dish/eval_video/2025.09.14-18.07.10"
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

EP_PAT = re.compile(r"^episode_(\d+)$")

def find_episode_dirs(root):
    for name in os.listdir(root):
        m = EP_PAT.match(name)
        if m:
            p = os.path.join(root, name)
            if os.path.isdir(p):
                yield int(m.group(1)), p

def load_arrays(ep_dir):
    gt_path = os.path.join(ep_dir, "gt.npy")
    sm_path = os.path.join(ep_dir, "smoothed.npy")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"missing {gt_path}")
    if not os.path.isfile(sm_path):
        raise FileNotFoundError(f"missing {sm_path}")
    gt = np.load(gt_path, allow_pickle=False)
    sm = np.load(sm_path, allow_pickle=False)
    return gt, sm

def mse(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    return float(np.mean((a[:n] - b[:n]) ** 2))

def main():
    results = []
    skipped = []

    for ep_num, ep_dir in find_episode_dirs(ROOT):
        try:
            gt, sm = load_arrays(ep_dir)
            loss = mse(gt, sm)
            if np.isnan(loss):
                skipped.append((ep_num, "empty arrays after alignment"))
                continue
            results.append((ep_num, loss))
        except Exception as e:
            skipped.append((ep_num, str(e)))

    # sort by loss (ascending)
    results.sort(key=lambda x: x[1])

    # print (episode_num, loss) pairs
    for ep, loss in results:
        print(f"{ep}\t{loss:.6g}")

    # optional summary
    if skipped:
        print(f"\nSkipped {len(skipped)} episodes:")
        for ep, reason in sorted(skipped):
            print(f"  {ep}: {reason}")

if __name__ == "__main__":
    main()
