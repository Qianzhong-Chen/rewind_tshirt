import torch
from typing import Callable
from pathlib import Path
from .lerobot_dataset import LeRobotDataset


class XdofLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        horizon: int = 1,
        root: str | Path | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        """
        XdofLeRobotDataset extends the LeRobotDataset to support sequence-based sampling for reinforcement learning or
        imitation learning tasks that require a history of observations and future action sequences.

        Key Features:
        -------------
        1. Sequence Support:
        This dataset class enables sampling of temporally contiguous sequences from raw episode-parquet datasets.
        Each item returned corresponds to a slice of a trajectory containing:
            - A fixed number of past observations (`n_obs_steps`)
            - A sequence of actions (and optionally states or other data) over a planning horizon (`horizon`)

        2. Time-Aligned Video Frame Querying:
        When available, RGB camera frames corresponding to the observation timestamps are queried and returned
        as tensors of shape [n_obs_steps, 3, H, W]. Padding is performed if the sequence is truncated near
        the episode boundaries.

        3. Padding for Truncated Sequences:
        Near episode boundaries, the available window may be shorter than the desired length. In such cases,
        the last available frame is duplicated (padded) to maintain consistent tensor shapes for batch loading.

        4. Compatibility:
        Fully compatible with the HuggingFace datasets used in LeRobot. Reuses the indexing mechanism
        (`episode_data_index`) to identify episode boundaries and ensures returned data does not cross them.

        5. Transform Support:
        Optional `image_transforms` are applied per camera frame after video querying, allowing
        on-the-fly preprocessing (e.g., resizing, normalization, augmentation).

        Use Cases:
        ----------
        - Multi-step policy learning (e.g., decision transformers, behavior cloning with horizon > 1)
        - Multi-frame vision inputs (e.g., stacking RGB frames for temporal context)
        - Evaluation pipelines where temporal consistency is required
        - Data preprocessing for autoregressive models

        Constructor Arguments:
        ----------------------
        - repo_id: Hugging Face repository ID containing the dataset.
        - episodes: Optional list of episode indices to restrict loading to a subset.
        - n_obs_steps: Number of past observations to include as context.
        - horizon: Number of future steps to predict (or simulate).
        - Other arguments are passed directly to LeRobotDataset and support caching, video download, and transformation.

        Returned Format (Example):
        --------------------------
        {
            "state": Tensor[n_obs_steps, ...],
            "actions": Tensor[horizon, ...],
            "left_camera-images-rgb": Tensor[n_obs_steps, 3, H, W],
            "right_camera-images-rgb": Tensor[n_obs_steps, 3, H, W],
            "task": string,
            ...
        }
        """
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
        )

        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.timestamp_tensor = torch.tensor(self.hf_dataset["timestamp"]).flatten()

    # add sequence support
    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        assert ep_idx in self.episodes, f"Episode {ep_idx} not found in the selected episodes."
        global_idx = self.episodes.index(ep_idx)
        
        ep_start = self.episode_data_index["from"][global_idx].item()
        ep_end = self.episode_data_index["to"][global_idx].item()
        
        # Define full sequence range: [t - n_obs_steps, ..., t + H - n_obs_steps - 1]
        window_start = max(ep_start, idx - self.n_obs_steps)
        window_end = min(ep_end, window_start + self.horizon)
        if window_end - window_start < self.horizon:
            window_start = max(ep_start, window_end - self.horizon)
        actual_len = window_end - window_start

        try:
            sequence = self.hf_dataset.select(range(window_start, window_end))
        except IndexError:
            sequence = self.hf_dataset.select([min(idx, len(self.hf_dataset)-1)])

        if actual_len < self.horizon:
            pad_count = self.horizon - actual_len
            pad_frame = {k: sequence[k][-1] for k in sequence.features}
            for k in sequence.features:
                sequence[k] += [pad_frame[k]] * pad_count
            del pad_frame

        # Extract correctly aligned sequences
        seq_item = {}
        for key in sequence.features:
            value = sequence[key]

            if key == "actions":
                # actions[t - n_obs_steps : t + H - n_obs_steps]
                seq_item[key] = torch.stack(value)
            elif key == "state":
                # obs[t - n_obs_steps : t]
                seq_item[key] = torch.stack(value[:self.n_obs_steps])
            else:
                seq_item[key] = value[0]
            del value

        del sequence

        # Query video frames
        obs_ts_range = self.timestamp_tensor[window_start : window_start + self.n_obs_steps].tolist()
        query_ts_dict = {key: obs_ts_range for key in self.meta.video_keys}
        video_frames = self._query_videos(query_ts_dict, ep_idx)
        for key in self.meta.video_keys:
            frames = video_frames[key]  # [T, 3, H, W] or [1, 3, H, W] if fallback

            if frames.shape[0] < self.n_obs_steps:
                pad_count = self.n_obs_steps - frames.shape[0]
                pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
                frames = torch.cat([frames, pad_frame], dim=0)

            seq_item[key] = frames  # [n_obs_steps, 3, H, W]
        

        # Clean up
        del video_frames, query_ts_dict, obs_ts_range

        # Apply image transforms
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in seq_item:
                    seq_item[cam] = self.image_transforms(seq_item[cam])

        task_idx = item["task_index"].item()
        seq_item["task"] = self.meta.tasks[task_idx]
        del item

        return seq_item

    
if __name__ == "__main__":
    # dataset = XdofLeRobotDataset(repo_id=cfg.task.repo_id, horizon=cfg.horizon, episodes=train_episodes, n_obs_steps=cfg.n_obs_steps)
    dataset = XdofLeRobotDataset(
        repo_id="Qianzhong-Chen/yam_pick_up_cube_sim_policy_pi0_joint_image_flip_new_0630",
        horizon=16,
        n_obs_steps=2
    )