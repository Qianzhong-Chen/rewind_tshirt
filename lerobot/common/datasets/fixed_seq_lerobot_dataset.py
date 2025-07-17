import torch
from typing import Callable
from pathlib import Path
from .lerobot_dataset import LeRobotDataset
import time
from typing import Tuple
from faker import Faker



class FixedSeqLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_obs_steps: int = 1,
        horizon: int = 1,
        max_rewind_steps: int = 0,
        root: str | Path | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        image_names: list[str] = ["top_camera-images-rgb"],
    ):
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
        self.max_rewind_steps = max_rewind_steps
        self.timestamp_tensor = torch.tensor(self.hf_dataset["timestamp"]).flatten()
        assert all(img_name in self.meta.video_keys for img_name in image_names), f"Image names {image_names} not found in metadata video keys."
        self.wrapped_video_keys = image_names  # Use only the specified camera for videos
        self.verbs = ['move', 'grasp', 'rotate', 'push', 'pull', 'slide', 'lift', 'place']
        self.fake = Faker()

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
        rewind_flag = False
        if idx > window_start:
            # randomly choose if rewind with 80% possibility
            rewind_flag = torch.rand(1).item() < 0.8

        
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
                seq_item[key] = torch.stack(value)
            elif key == "state":
                seq_item[key] = torch.stack(value[:self.n_obs_steps])
            else:
                seq_item[key] = value[0]
            del value

        del sequence

        # Query video frames
        obs_ts_range = self.timestamp_tensor[window_start : window_start + self.n_obs_steps].tolist()
        query_ts_dict = {key: obs_ts_range for key in self.wrapped_video_keys}
        video_frames = self._query_videos(query_ts_dict, ep_idx)
        rewind_step = None
        for key in self.wrapped_video_keys:
            frames = video_frames[key]  # [T, 3, H, W] or [1, 3, H, W] if fallback

            if frames.shape[0] < self.n_obs_steps:
                pad_count = self.n_obs_steps - frames.shape[0]
                pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
                frames = torch.cat([frames, pad_frame], dim=0)

            if rewind_flag:
                rewind_step, rewind_frames = self._get_rewind(idx, window_start, key, ep_idx, rewind_step=rewind_step)
                frames = torch.cat([frames, rewind_frames], dim=0)
            else:
                rewind_step = 0
                padding_frames = torch.zeros((self.max_rewind_steps, *frames.shape[1:]), dtype=frames.dtype)
                frames = torch.cat([frames, padding_frames], dim=0)
            seq_item[key] = frames  # [n_obs_steps, 3, H, W]
        
        # Clean up
        del video_frames, query_ts_dict, obs_ts_range

        # Apply image transforms
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in seq_item:
                    seq_item[cam] = self.image_transforms(seq_item[cam])

        pertube_task_flag = torch.rand(1).item() < 0.2
        if pertube_task_flag:
            num_words = torch.randint(1, 6, (1,)).item()
            verb = self.verbs[torch.randint(0, len(self.verbs), (1,)).item()]
            phrase = [verb] + self.fake.words(nb=num_words)
            phrase = " ".join(phrase)
            seq_item["task"] = phrase
        else:
            task_idx = item["task_index"].item()
            # seq_item["task"] = self.meta.tasks[task_idx] 
            # TODO: dataset error, use temorary fix
            seq_item["task"] = "pick up the cube"
        del item 

        progress_start = (window_start - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
        progress_end = (idx - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
        progress_list = torch.linspace(progress_start, progress_end, self.n_obs_steps, dtype=torch.float32)
        seq_item["targets"] = torch.zeros(self.n_obs_steps + self.max_rewind_steps, dtype=torch.float32)

        if not pertube_task_flag:
            seq_item["targets"][:self.n_obs_steps] = progress_list
            for i in range(rewind_step):
                seq_item["targets"][self.n_obs_steps + i] = torch.flip(progress_list, dims=[0])[i+1]
        
        seq_item["lengths"] = torch.tensor(self.n_obs_steps + rewind_step, dtype=torch.int32)

        del progress_start, progress_end, progress_list
        return seq_item


    def _get_rewind(self, idx: int, window_start: int, key: str, ep_idx: int, rewind_step=None) -> Tuple[int, torch.tensor]:
        assert self.max_rewind_steps < self.n_obs_steps, "Max rewind steps must be less than n_obs_steps."
        assert idx > window_start, f"Index {idx} must be greater than window start {window_start} for rewind."
        # print(idx, window_start)
        rewind_step_limit = max(min(self.max_rewind_steps, idx - window_start), 1)
        if rewind_step is None:
            # Randomly choose a rewind step
            rewind_step = torch.randint(1, rewind_step_limit + 1, (1,)).item()
        rewind_ts_range = self.timestamp_tensor[idx - rewind_step : idx].tolist()
        query_ts_dict = {key: rewind_ts_range}
        rewind_frames = self._query_videos(query_ts_dict, ep_idx)[key]
        
        if rewind_step == 1:
            rewind_frames = rewind_frames.unsqueeze(0)
        rewind_frames = torch.flip(rewind_frames, dims=[0])

        padding_needed = self.max_rewind_steps - rewind_step
        if padding_needed > 0:
            # Pad with the last frame if needed
            last_frame = torch.zeros_like(rewind_frames[0])
            padding_frames = torch.zeros((padding_needed, *last_frame.shape), dtype=rewind_frames.dtype)
            rewind_frames = torch.cat([rewind_frames, padding_frames], dim=0)

        return rewind_step, rewind_frames
