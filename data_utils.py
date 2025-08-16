import os
import re
from pathlib import Path
from typing import List, Union, Tuple
import random

def comply_lerobot_batch(batch: dict, camera_names: List[str] = ["top_camera-images-rgb"]) -> dict:
        """Comply with lerobot dataset batch format."""
        # convert to diffusion dataset format
        # this is a hack to make it work with lerobot dataset
        result =  {
            "image_frames": {},
            "targets": batch["targets"],
            "lengths": batch["lengths"],
            "tasks": batch["task"],
        }

        for cam_name in camera_names:
            result["image_frames"][cam_name] = batch[cam_name]

        return result

def comply_lerobot_batch_multi_stage(batch: dict, camera_names: List[str] = ["top_camera-images-rgb"], dense_annotation: bool = False) -> dict:
        """Comply with lerobot dataset batch format."""
        # convert to diffusion dataset format
        # this is a hack to make it work with lerobot dataset
        result =  {
            "image_frames": {},
            "targets": batch["targets"],
            "lengths": batch["lengths"],
            "tasks": batch["task"],
            "state": batch["state"],
            "frame_relative_indices": batch["frame_relative_indices"],
        }

        for cam_name in camera_names:
            result["image_frames"][cam_name] = batch[cam_name]

        if dense_annotation:
            transposed = list(map(list, zip(*result["tasks"])))
            result["tasks"] = transposed

        return result
    
def comply_lerobot_batch_multi_stage_hybird(batch: dict, 
                                            camera_names: List[str] = ["top_camera-images-rgb"], 
                                            dense_annotation: bool = False,
                                            anno_type: str = "sparse") -> dict:
        """Comply with lerobot dataset batch format."""
        # convert to diffusion dataset format
        # this is a hack to make it work with lerobot dataset
        
        result =  {
            "image_frames": {},
            "targets": batch["targets"],
            "lengths": batch["lengths"],
            "tasks": batch["task"],
            "state": batch["state"],
            "frame_relative_indices": batch["frame_relative_indices"],
        }
        
        if "anno_type" in batch:
            result["anno_type"] = batch["anno_type"]
        else:
            result["anno_type"] = anno_type

        for cam_name in camera_names:
            result["image_frames"][cam_name] = batch[cam_name]

        if dense_annotation:
            transposed = list(map(list, zip(*result["tasks"])))
            result["tasks"] = transposed

        return result


def comply_lerobot_batch_multi_stage_video_eval(batch: dict, camera_names: List[str] = ["top_camera-images-rgb"], dense_annotation: bool = False) -> dict:
        """Comply with lerobot dataset batch format."""
        # convert to diffusion dataset format
        # this is a hack to make it work with lerobot dataset
        result =  {
            "image_frames": {},
            "targets": batch["targets"].unsqueeze(0),  
            "lengths": batch["lengths"].unsqueeze(0),
            "tasks": [batch["task"]],
            "state": batch["state"].unsqueeze(0),
            "frame_relative_indices": batch["frame_relative_indices"].unsqueeze(0),
        }

        for cam_name in camera_names:
            result["image_frames"][cam_name] = batch[cam_name].unsqueeze(0)

        return result


def get_valid_episodes(repo_id: str) -> List[int]:
    """
    Collects valid episode indices under the lerobot cache for the given repo_id.

    Args:
        repo_id (str): HuggingFace repo ID, e.g., 'Qianzhong-Chen/yam_pick_up_cube_sim_rotate_0704'

    Returns:
        List[int]: Sorted list of valid episode indices (e.g., [0, 1, 5, 7, ...])
    """
    base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"
    episode_pattern = re.compile(r"episode_(\d+)\.parquet")

    valid_episodes = []

    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    for chunk_dir in base_path.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        for file in chunk_dir.glob("episode_*.parquet"):
            match = episode_pattern.match(file.name)
            if match:
                ep_idx = int(match.group(1))
                valid_episodes.append(ep_idx)

    return sorted(valid_episodes)

def split_train_eval_episodes(valid_episodes: List[int], train_ratio: float = 0.9, seed: int = 42) -> Tuple[List[int], List[int]]:
    """
    Randomly split valid episodes into training and evaluation sets.

    Args:
        valid_episodes (List[int]): List of valid episode indices.
        train_ratio (float): Fraction of episodes to use for training (default: 0.9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[List[int], List[int]]: (train_episodes, eval_episodes)
    """
    random.seed(seed)
    episodes = valid_episodes.copy()
    random.shuffle(episodes)

    split_index = int(len(episodes) * train_ratio)
    train_episodes = episodes[:split_index]
    eval_episodes = episodes[split_index:]

    return train_episodes, eval_episodes

if __name__ == "__main__":
    repo_id = "Qianzhong-Chen/yam_pick_up_cube_sim_policy_pi0_joint_image_flip_new_0630"
    valid_episodes = get_valid_episodes(repo_id)
    print(f"Valid episodes for {repo_id}: {valid_episodes}")
    train_episodes, eval_episodes = split_train_eval_episodes(valid_episodes)
    print(f"Train episodes: {train_episodes}")
    print(f"Eval episodes: {eval_episodes}")
