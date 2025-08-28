import torch
from typing import Callable
from pathlib import Path
from .lerobot_dataset import LeRobotDataset
import time
from typing import Tuple
from faker import Faker
import random



class EpsLeRobotDataset(LeRobotDataset):
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
        image_names: list[str] = ["top_camera-images-rgb"],
        dense_annotation: bool = False,
        video_eval: bool = False,
        annotation_list: list[str] | None = None,
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
        self.timestamp_tensor = torch.tensor(self.hf_dataset["timestamp"]).flatten()
        assert all(img_name in self.meta.video_keys for img_name in image_names), f"Image names {image_names} not found in metadata video keys."
        self.wrapped_video_keys = image_names  # Use only the specified camera for videos
        self.verbs = ['move', 'grasp', 'rotate', 'push', 'pull', 'slide', 'lift', 'place']
        self.fake = Faker()
        self.dense_annotation = dense_annotation
        self.video_eval = video_eval
        self.annotation_list = annotation_list

    # add fixed ep start to sequence
    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        assert ep_idx in self.episodes, f"Episode {ep_idx} not found in the selected episodes."
        global_idx = self.episodes.index(ep_idx)

        ep_start = self.episode_data_index["from"][global_idx].item()
        ep_end = min(self.episode_data_index["to"][global_idx].item(), len(self)-1)

        num_frames = max(0, ep_end - ep_start + 1)

        if num_frames <= 0:
            raise ValueError(f"Empty episode: start={ep_start}, end={ep_end}")

        target_len = self.n_obs_steps + 1  # always include frame 0 + n_obs_steps frames

        if num_frames >= target_len:
            # Evenly sample indices from [0, num_frames-1], inclusive, length = target_len
            # Always starts at 0 and ends at num_frames-1
            step = (num_frames - 1) / (target_len - 1)
            rel = [int(round(k * step)) for k in range(target_len)]  # monotonic nondecreasing
            obs_indices = [ep_start + r for r in rel]
        else:
            # Not enough frames: take all frames, then pad with the last frame
            all_frames = list(range(ep_start, ep_end + 1))
            pad_count = target_len - num_frames
            obs_indices = all_frames + [ep_end] * pad_count

        shuffled = obs_indices[1:]
        random.shuffle(shuffled)
        obs_indices[1:] = shuffled

        # Now fetch the mini-sequence
        sequence = self.hf_dataset.select(obs_indices)

        seq_item = {}
        for key in sequence.features:
            value = sequence[key]
            if key in ("actions", "state"):
                seq_item[key] = torch.stack(value)  # shape: (target_len, ...)
            elif key == "reward":
                progress_list = torch.stack(value).squeeze(-1)  # (target_len,)
            else:
                # If this is per-episode metadata, keep the first;
                # if it's actually per-frame (e.g., images), consider stacking instead.
                seq_item[key] = value[0]
            del value
        del sequence

        # Query video frames
        obs_ts_range = self.timestamp_tensor[obs_indices].tolist()
        query_ts_dict = {key: obs_ts_range for key in self.wrapped_video_keys}
        video_frames = self._query_videos(query_ts_dict, ep_idx)
        
        
        for key in self.wrapped_video_keys:
            frames = video_frames[key]
            if frames.shape[0] < self.n_obs_steps:
                pad_count = self.n_obs_steps - frames.shape[0]
                pad_frame = frames[-1:].repeat(pad_count, 1, 1, 1)
                frames = torch.cat([frames, pad_frame], dim=0)

            seq_item[key] = frames

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in seq_item:
                    seq_item[cam] = self.image_transforms(seq_item[cam])

        # Task string
        pertube_task_flag = torch.rand(1).item() < 0.2
        if self.video_eval:
            pertube_task_flag = False
        if pertube_task_flag:
            num_words = torch.randint(1, 6, (1,)).item()
            verb = self.verbs[torch.randint(0, len(self.verbs), (1,)).item()]
            phrase = [verb] + self.fake.words(nb=num_words)
            seq_item["task"] = " ".join(phrase)
        else:
            seq_item["task"] = "fold the tshirt"

        # Progress targets
        seq_item["targets"] = torch.zeros(1 + self.n_obs_steps, dtype=torch.float32)
        frame_relative_indices = torch.zeros(1 + self.n_obs_steps, dtype=torch.float32)
        for i, idx in enumerate(obs_indices):
            frame_relative_indices[i] = (idx - ep_start) / (ep_end - ep_start) if ep_end > ep_start else 0.0
        seq_item["frame_relative_indices"] = frame_relative_indices
        
        if not pertube_task_flag:
            seq_item["targets"][:self.n_obs_steps + 1] = progress_list
            
        
        seq_item["state"] = seq_item["state"]
        seq_item["lengths"] = torch.tensor(1 + self.n_obs_steps, dtype=torch.int32)

        if self.dense_annotation:
            if pertube_task_flag:
                seq_item["task"] = [seq_item["task"]] * (1 + self.n_obs_steps)
            else:
                seq_item["task"] = [''] * (1 + self.n_obs_steps)
                # Five-staged task annotation
                for i in range(0, 1 + self.n_obs_steps ):
                    stage_idx =  int(torch.floor(seq_item["targets"][i]).item())
                    stage_idx = min(stage_idx, len(self.annotation_list) - 1)
                    seq_item["task"][i] = self.annotation_list[stage_idx]

        del item, video_frames, query_ts_dict, obs_ts_range, progress_list

        return seq_item



if __name__ == "__main__":
    from data_utils import get_valid_episodes
    
    repo_id = "Qianzhong-Chen/tshirt_reward_yam_only_multi_0804"
    valid_episodes = get_valid_episodes(repo_id)
