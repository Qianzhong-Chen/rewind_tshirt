import torch
from typing import Callable
from pathlib import Path
from .lerobot_dataset import LeRobotDataset
from faker import Faker


class DynamicSeqLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        episodes: list[int] | None = None,
        n_seq: int = 10,
        max_rewind_steps: int = 4,
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
        self.n_seq = n_seq
        self.max_rewind_steps = max_rewind_steps
        self.timestamp_tensor = torch.tensor(self.hf_dataset["timestamp"]).flatten()
        assert all(img_name in self.meta.video_keys for img_name in image_names), f"Image names {image_names} not found."
        self.wrapped_video_keys = image_names
        self.verbs = ['move', 'grasp', 'rotate', 'push', 'pull', 'slide', 'lift', 'place']
        self.fake = Faker()

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        assert ep_idx in self.episodes
        global_idx = self.episodes.index(ep_idx)
        ep_start = self.episode_data_index["from"][global_idx].item()
        ep_end = self.episode_data_index["to"][global_idx].item()
        ep_len = ep_end - ep_start

        # Downsample episode to n_seq key frames
        if ep_len < self.n_seq:
            key_indices = list(range(ep_start, ep_end))
        else:
            key_indices = torch.linspace(ep_start, ep_end - 1, self.n_seq).round().long().tolist()

        # Find nearest key frame to idx
        closest_idx = min(range(len(key_indices)), key=lambda i: abs(key_indices[i] - idx))
        rewind_flag = torch.rand(1).item() < 0.8

        # Determine rewind step count
        current_max_rewind = min(closest_idx, self.max_rewind_steps)
        rewind_step = torch.randint(1, current_max_rewind + 1, (1,)).item() if rewind_flag and current_max_rewind > 0 else 0

        # Get sequence indices and rewind indices
        final_indices = key_indices
        if len(final_indices) < self.n_seq:
            pad_count = self.n_seq - len(final_indices)
            final_indices += [final_indices[-1]] * pad_count

        try:
            sequence = self.hf_dataset.select(final_indices)
        except IndexError:
            sequence = self.hf_dataset.select([final_indices[-1]])

        seq_item = {}
        for key in sequence.features:
            if key == "actions":
                seq_item[key] = torch.stack(sequence[key])
            elif key == "state":
                seq_item[key] = torch.stack(sequence[key])
            else:
                seq_item[key] = sequence[key][0]

        # Query video frames
        n_forward = closest_idx + 1
        forward_indices = key_indices[:n_forward]

        for cam in self.wrapped_video_keys:
            forward_ts = [self.timestamp_tensor[i].item() for i in forward_indices]
            query_ts_dict = {cam: forward_ts}
            forward_frames = self._query_videos(query_ts_dict, ep_idx)[cam]
            if forward_frames.ndim == 3:
                forward_frames = forward_frames.unsqueeze(0)

            # Rewind handling
            if rewind_step > 0:
                rewind_frames = self._get_rewind(key_indices, closest_idx, rewind_step, cam, ep_idx)
            else:
                rewind_frames = torch.zeros((0, *forward_frames.shape[1:]), dtype=forward_frames.dtype)

            # Final padding
            n_total = n_forward + rewind_frames.shape[0]
            n_extra_pad = self.n_seq + self.max_rewind_steps - n_total
            padding_frames = torch.zeros((n_extra_pad, *forward_frames.shape[1:]), dtype=forward_frames.dtype)

            full_frames = torch.cat([forward_frames, rewind_frames, padding_frames], dim=0)
            seq_item[cam] = full_frames  # shape: [<=n_seq + max_rewind_steps, C, H, W]

        # print(full_frames.shape,)
        if len(full_frames.shape) == 3:
            print(forward_frames.shape, rewind_frames.shape, padding_frames.shape)
        # Apply image transforms
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in seq_item:
                    seq_item[cam] = self.image_transforms(seq_item[cam])

        # Task prompt
        pertube_task_flag = False
        if torch.rand(1).item() < 0.2:
            pertube_task_flag = True
            num_words = torch.randint(1, 6, (1,)).item()
            verb = self.verbs[torch.randint(0, len(self.verbs), (1,)).item()]
            phrase = [verb] + self.fake.words(nb=num_words)
            seq_item["task"] = " ".join(phrase)
        else:
            seq_item["task"] = "pick up the cube"

        # Episode-relative progress
        progress = torch.tensor(
            [(i - ep_start) / (ep_end - ep_start) for i in forward_indices],
            dtype=torch.float32
        )

        targets = torch.zeros(self.n_seq + self.max_rewind_steps, dtype=torch.float32)
        targets[:n_forward] = progress

        if rewind_step > 0:
            rewind_targets = torch.flip(progress, dims=[0])[1:rewind_step + 1]
            targets[n_forward:n_forward + rewind_step] = rewind_targets
        if not pertube_task_flag:
            seq_item["targets"] = targets
        else:
            seq_item["targets"] = torch.zeros_like(targets, dtype=torch.float32)
        seq_item["lengths"] = torch.tensor(n_forward + rewind_step, dtype=torch.int32)

        return seq_item


    def _get_rewind(self, key_indices, current_idx, rewind_step, key, ep_idx):
        assert rewind_step > 0
        start_idx = key_indices[current_idx - rewind_step : current_idx]
        rewind_ts = [self.timestamp_tensor[i].item() for i in start_idx]
        query_ts_dict = {key: rewind_ts}
        rewind_frames = self._query_videos(query_ts_dict, ep_idx)[key]

        # Ensure shape is (T, C, H, W)
        if rewind_frames.ndim == 3:
            # single frame [C, H, W] → [1, C, H, W]
            rewind_frames = rewind_frames.unsqueeze(0)
        elif rewind_frames.ndim == 4 and rewind_frames.shape[1] not in [1, 3]:
            # Likely [T, H, W] — expand channels
            rewind_frames = rewind_frames.unsqueeze(1)  # → [T, 1, H, W]
            rewind_frames = rewind_frames.repeat(1, 3, 1, 1)  # fake RGB if needed

        rewind_frames = torch.flip(rewind_frames, dims=[0])

        # Padding if needed
        n_pad = self.max_rewind_steps - rewind_step
        if n_pad > 0:
            pad = torch.zeros((n_pad, *rewind_frames.shape[1:]), dtype=rewind_frames.dtype)
            rewind_frames = torch.cat([rewind_frames, pad], dim=0)

        return rewind_frames

