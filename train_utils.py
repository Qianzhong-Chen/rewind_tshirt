import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from normalizer import SingleFieldLinearNormalizer
import json
import numpy as np

def set_seed(s): random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_ckpt(model, opt, ep, save_dir, input_name=None):
    save_dir = Path(save_dir) / "checkpoints"  # convert to Path first
    name = f"{input_name}.pt" if input_name else f"epoch{ep:04d}.pt"
    p = save_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(model=model.state_dict(),
             optimizer=opt.state_dict(),
             epoch=ep),
        p
    )


@torch.no_grad()
def get_normalizer_from_calculated(path, device) -> SingleFieldLinearNormalizer:
    with open(path, 'r') as f:
        norm_data = json.load(f)['norm_stats']

    def to_tensor_slice(data, k=14): # both arms
        return torch.tensor(data[:k], dtype=torch.float32).to(device)

    # Process state
    state_stats = norm_data['state']
    state_normalizer = SingleFieldLinearNormalizer.create_manual(
        scale=1. / to_tensor_slice(state_stats['std']),
        offset=-to_tensor_slice(state_stats['mean']) / to_tensor_slice(state_stats['std']),
        input_stats_dict={
            'min': to_tensor_slice(state_stats['q01']),
            'max': to_tensor_slice(state_stats['q99']),
            'mean': to_tensor_slice(state_stats['mean']),
            'std': to_tensor_slice(state_stats['std']),
        }
    )

    return state_normalizer

def plot_pred_vs_gt(pred: torch.Tensor, gt: torch.Tensor, indices: torch.Tensor, save_path: Path):
    """
    Plot predicted vs ground truth reward using scatter points over real and discrete timesteps.
    Rewind points are where GT value repeats a previous non-zero value, shown as hollow markers.

    Args:
        pred (torch.Tensor): Predicted reward, shape (T,) or (1, T)
        gt (torch.Tensor): Ground truth reward, shape (T,) or (1, T)
        indices (torch.Tensor): Normalized timesteps in [0, 1], shape (T,)
        save_path (Path): Base path to save plots (e.g., result_dir / "plot.png")
    """
    # Ensure shape is (T,)
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()
    indices_np = indices.squeeze().cpu().numpy()
    timesteps = list(range(len(pred_np)))

    # --- Identify rewind indices ---
    seen = set()
    rewind_indices = []
    for i, val in enumerate(gt_np):
        if val == 0.0:
            continue
        if val in seen:
            rewind_indices.append(i)
        else:
            seen.add(val)
    rewind_indices = np.array(rewind_indices)
    regular_indices = np.setdiff1d(np.arange(len(gt_np)), rewind_indices)

    # === Plot 1: Real timesteps ===
    plt.figure()
    # Regular points
    plt.scatter(indices_np[regular_indices], pred_np[regular_indices], label="Predicted", marker='o')
    plt.scatter(indices_np[regular_indices], gt_np[regular_indices], label="Ground Truth", marker='x')
    # Hollow rewind points (overlayed)
    if len(rewind_indices) > 0:
        plt.scatter(indices_np[rewind_indices], pred_np[rewind_indices], 
                    label="Predicted_Rewind", marker='^', facecolors='none', edgecolors='blue', linewidths=1.5)
        plt.scatter(indices_np[rewind_indices], gt_np[rewind_indices], 
                    label="GT_Rewind", marker='s', facecolors='none', edgecolors='red', linewidths=1.5)

    plt.xlabel("Real Relative Timestep")
    plt.ylabel("Reward")
    plt.title("Predicted vs Ground Truth (Real Timestep)")
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=[round(x, 2) for x in indices_np], labels=[f"{x:.2f}" for x in indices_np], rotation=45)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path.with_name(save_path.stem + "_real.png")))
    plt.close()

    # === Plot 2: Discrete timesteps ===
    plt.figure()
    plt.scatter(timesteps, pred_np, label="Predicted", marker='o', color='blue')
    plt.plot(timesteps, pred_np, linestyle='-', alpha=0.7, color='blue')  # Line for predicted

    plt.scatter(timesteps, gt_np, label="Ground Truth", marker='x', color='orange')
    plt.plot(timesteps, gt_np, linestyle='--', alpha=0.7, color='orange')  # Line for GT

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Predicted vs Ground Truth (Discrete Timestep)")
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=timesteps)
    plt.tight_layout()
    plt.savefig(str(save_path.with_name(save_path.stem + "_discrete.png")))
    plt.close()

def plot_episode_result(ep_index, ep_result, gt_ep_result, x_offset, rollout_save_dir):
    save_dir = rollout_save_dir / f"episode_{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Trim initial frames
    ep_result = ep_result[x_offset:]
    gt_ep_result = gt_ep_result[x_offset:]

    # Convert to numpy arrays
    ep_result_np = np.array(ep_result)
    gt_ep_result_np = np.array(gt_ep_result)
    timestep = np.arange(len(ep_result_np)) + x_offset

    # Compute MSE and MAE
    mse = np.mean((ep_result_np - gt_ep_result_np) ** 2)
    mae = np.mean(np.abs(ep_result_np - gt_ep_result_np))

    # Plot
    plt.figure()
    plt.plot(timestep, ep_result_np, label="Predicted")
    plt.plot(timestep, gt_ep_result_np, label="Ground Truth")
    # Add dummy lines for metrics in the legend
    plt.plot([], [], ' ', label=f"MSE: {mse:.4f}")
    plt.plot([], [], ' ', label=f"MAE: {mae:.4f}")
    plt.title("Episode Result")
    plt.xlabel("Time Step")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "plot.png")
    plt.close()

    return str(save_dir)

def plot_episode_result_raw_data(ep_index, ep_result, x_offset, rollout_save_dir):
    save_dir = rollout_save_dir / f"episode_{ep_index}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Trim initial frames
    ep_result = ep_result[x_offset:]

    # Convert to numpy arrays
    ep_result_np = np.array(ep_result)
    timestep = np.arange(len(ep_result_np)) + x_offset

   
    # Plot
    plt.figure()
    plt.plot(timestep, ep_result_np, label="Predicted")
    # Add dummy lines for metrics in the legend
    plt.title("Episode Result")
    plt.xlabel("Time Step")
    plt.ylabel("Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / "plot.png")
    plt.close()

    return str(save_dir)