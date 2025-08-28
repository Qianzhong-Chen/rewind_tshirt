from pathlib import Path
from omegaconf import OmegaConf
import itertools
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import wandb
from lerobot.common.datasets.frame_gap_multi_stage_lerobot_dataset import FrameGapLeRobotDataset 
from data_utils import comply_lerobot_batch_regression, get_valid_episodes, split_train_eval_episodes, comply_lerobot_batch_multi_stage_video_eval
from train_utils import plot_episode_result, set_seed, save_ckpt, plot_pred_vs_gt, get_normalizer_from_calculated, plot_episode_result, plot_episode_result_raw_data
from raw_data_utils import get_frame_num, get_frame_data_fast, get_traj_data
from models.multi_stage_reward_net import RewardTransformer
from models.clip_encoder import FrozenCLIPEncoder
from make_demo_video import produce_video, produce_video_raw_data, produce_video_raw_data_hybird
import torch.nn as nn
import cv2
import numpy as np
import time
from torchvision.utils import save_image
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"

def vlc_loss(
    reward_seq: torch.Tensor,   # (B, T)
    lens: torch.Tensor,         # (B,), valid lengths (<= T)
    *,
    alpha: float = 1.0,         # weight on monotonic-ranking (set as you like)
    shift_to_zero: bool = False, # subtract r[:,0] so each episode starts at 0
    tv_beta: float = 0.0,       # optional smoothness weight (0 = off)
    margin: float = 0.0,        # optional desired minimum per-step increase
):
    """
    Row-independent loss (no cross-batch contrastive).
    Enforces monotonic increase within each sequence using a hinge on decreases.

    L = alpha * sum_t max( (r_t - r_{t+1}) + margin, 0 ) + tv_beta * smoothness

    Args:
        reward_seq: predicted rewards per (video, caption) pair up to each step, shape (B, T).
        lens:       valid lengths per sequence (<= T), shape (B,).
        alpha:      weight for the ranking term.
        shift_to_zero: if True, shift per-row so reward at t=0 is 0 (stabilizes scales).
        tv_beta:    weight on optional total-variation smoothness (second difference).
        margin:     non-negative margin; margin>0 encourages strictly increasing by `margin`.

    Returns:
        total_loss: scalar
        logs: dict of detached components
    """
    device = reward_seq.device
    B, T = reward_seq.shape

    r = reward_seq
    if shift_to_zero:
        r = r - r[:, :1]  # (B, T), subtract first step

    if T <= 1:
        # nothing to rank
        ranking = torch.zeros((), device=device)
        tv_loss = torch.zeros((), device=device)
        total = alpha * ranking + tv_beta * tv_loss
        return total, {"ranking": ranking.detach(), "tv": tv_loss.detach()}

    # --- valid mask for (t -> t+1) transitions ---
    # valid if t < lens[i]-1
    idx = torch.arange(T - 1, device=device).unsqueeze(0)           # (1, T-1)
    valid = idx < (lens.view(-1, 1) - 1).clamp_min(0)               # (B, T-1)

    # --- monotonic ranking: penalize decreases (and shortfalls vs margin) ---
    # dec = (r_t - r_{t+1}) + margin  -> penalize positive part
    dec = r[:, :-1] - r[:, 1:] + margin                              # (B, T-1)
    rank_term = F.relu(dec)                                          # hinge
    # average over valid transitions only
    denom = valid.sum().clamp_min(1)
    ranking = (rank_term * valid).sum() / denom

    # --- optional smoothness (2nd-order total variation) ---
    if tv_beta > 0.0 and T >= 3:
        # r_{t+1} - 2 r_t + r_{t-1}
        second_diff = r[:, 2:] - 2 * r[:, 1:-1] + r[:, :-2]          # (B, T-2)
        # valid mask for second differences: t in [1 .. lens-2]
        idx2 = torch.arange(T - 2, device=device).unsqueeze(0)
        valid2 = idx2 < (lens.view(-1, 1) - 2).clamp_min(0)
        tv_loss = (second_diff.pow(2) * valid2).sum() / valid2.sum().clamp_min(1)
    else:
        tv_loss = torch.zeros((), device=device)

    total = alpha * ranking + tv_beta * tv_loss
    return total, {"total_loss": total.item(), "ranking": ranking.item(), "tv": tv_loss.item()}

class RewindRewardWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)
        self.camera_names = cfg.general.camera_names
        self.save_dir = Path(f'{cfg.general.project_name}/{cfg.general.task_name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Logging & ckpts to: {self.save_dir}")

    def train(self):
        cfg = self.cfg
        OmegaConf.save(cfg, self.save_dir / "config.yaml")
        # --- wandb ---
        wandb.init(
            project=f'{cfg.general.project_name}-{cfg.general.task_name}',
            name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
            config=cfg,
        )

        # --- data ---
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)

        dataset_train = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               horizon=cfg.model.horizon, 
                                               episodes=train_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               dense_annotation=cfg.model.dense_annotation,
                                               annotation_list=cfg.model.annotation_list)

        dataset_val = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               horizon=cfg.model.horizon, 
                                               episodes=val_eps, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               dense_annotation=cfg.model.dense_annotation,
                                               annotation_list=cfg.model.annotation_list)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg.dataloader)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val, **cfg.val_dataloader)
        dataloader_rollout = torch.utils.data.DataLoader(dataset_val, **cfg.rollout_dataloader)
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # --- encoders ---
        # # DINO
        # vis_encoder = FrozenVisionEncoder(cfg.encoders.vision_ckpt, self.device)
        # text_encoder = FrozenTextEncoder(cfg.encoders.text_ckpt, self.device)
        # vis_dim = vis_encoder.model.config.hidden_size
        # txt_dim = text_encoder.model.config.hidden_size

        # CLIP
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_encoder = clip_encoder
        text_encoder = clip_encoder
        vis_dim = 512
        txt_dim = 512


        # --- reward_model ---
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  dense_annotation=cfg.model.dense_annotation).to(self.device)
        

        if cfg.model.resume_training:
            reward_model_path = Path(cfg.model.model_path)
            # Load checkpoints
            reward_ckpt = torch.load(reward_model_path, map_location=self.device)
            # Load weights
            reward_model.load_state_dict(reward_ckpt["model"])
            # Move to device
            reward_model.to(self.device)
            reward_model.train()


        # Optimizer
        reward_optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        
        # Schedulers
        # Reward scheduler
        reward_warmup_scheduler = LinearLR(
            reward_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  # or 0.0 for full ramp-up
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        reward_cosine_scheduler = CosineAnnealingLR(
            reward_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,  # cosine decay after warmup
            eta_min=0.0  # or set a nonzero final LR if needed
        )
        reward_scheduler = SequentialLR(
            reward_optimizer,
            schedulers=[reward_warmup_scheduler, reward_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )

        
        best_val = float("inf")
        step = 0
        for epoch in range(1, cfg.train.num_epochs + 1):
            reward_model.train()
            with tqdm(dataloader_train, desc=f"Epoch {epoch}") as pbar:
                for batch in pbar:
                    batch = comply_lerobot_batch_regression(batch, 
                                                            camera_names=cfg.general.camera_names, 
                                                            dense_annotation=cfg.model.dense_annotation)
                    
                    B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                    img_list = []
                    for key in self.camera_names:
                        imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                        img_list.append(imgs)
                    
                    lang_strs = batch["tasks"]
                    trg = batch["targets"].to(self.device)
                    lens = batch["lengths"].to(self.device)
                    state = batch["state"].to(self.device)
                    
                    with torch.no_grad():
                        state = state_normalizer.normalize(state)
                        # CLIP
                        # img_emb = [clip_encoder.encode_image(imgs).view(B, T, -1) for imgs in img_list]
                        # img_emb = torch.stack(img_emb, dim=1)
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        if cfg.model.dense_annotation:
                            lang_emb = torch.zeros((B, T, txt_dim), dtype=torch.float32, device=self.device)  # (B, T, txt_dim)
                            for i in range(B):
                                lang_emb[i, :, :] = clip_encoder.encode_text(lang_strs[i])
                        else:
                            lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                    if cfg.model.no_state:
                        state = torch.zeros_like(state, device=self.device)
                    reward_pred = reward_model(img_emb, lang_emb, state, lens)
                    reward_loss, info = vlc_loss(reward_pred, lens)

                    reward_optimizer.zero_grad()
                    reward_loss.backward()
                    reward_unclipped = nn.utils.clip_grad_norm_(reward_model.parameters(), float("inf")).item()
                    _ = nn.utils.clip_grad_norm_(reward_model.parameters(), cfg.train.grad_clip)
                    reward_optimizer.step()
                    reward_scheduler.step()

                    
                    if step % cfg.train.log_every == 0:
                        train_info = {}
                        for k, v in info.items():
                            train_info[f"train/{k}"] = v
                        wandb.log({
                            **train_info,
                            "train/lr": reward_scheduler.get_last_lr()[0],
                            "train/reward_grad_norm": reward_unclipped,
                            "epoch": epoch,
                        }, step=step)
                    
                    pbar.set_postfix(loss=f"{(reward_loss.item()):.4f}")

                    if step % cfg.train.save_every == 0:
                        save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name=f"reward_step_{step:06d}_loss_{reward_loss.item():.3f}")
                    step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                reward_model.eval()
                total_loss, num = 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    for batch in dataloader_val:
                        batch = comply_lerobot_batch_regression(batch, 
                                                         camera_names=cfg.general.camera_names, 
                                                         dense_annotation=cfg.model.dense_annotation)
                        B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                        img_list = []
                        for key in self.camera_names:
                            imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                            img_list.append(imgs)
                        
                        lang_strs = batch["tasks"]
                        trg = batch["targets"].to(self.device)
                        lens = batch["lengths"].to(self.device)
                        state = batch["state"].to(self.device)
                        state = state_normalizer.normalize(state)

                        # CLIP
                        # img_emb = [clip_encoder.encode_image(imgs).view(B, T, -1) for imgs in img_list]
                        # img_emb = torch.stack(img_emb, dim=1)
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        if cfg.model.dense_annotation:
                            lang_emb = torch.zeros((B, T, txt_dim), dtype=torch.float32, device=self.device)  # (B, T, txt_dim)
                            for i in range(B):
                                lang_emb[i, :, :] = clip_encoder.encode_text(lang_strs[i])
                        else:
                            lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                        if cfg.model.no_state:
                            state = torch.zeros_like(state, device=self.device)
                        reward_pred = reward_model(img_emb, lang_emb, state, lens)
                        reward_loss, info = vlc_loss(reward_pred, lens)
                        total_loss += reward_loss.item()
                        num += 1

                val_loss = total_loss / num 
                print(f"[Eval] Epoch {epoch} Val L1: {val_loss:.6f}")
                wandb.log({"val/loss": val_loss}, step=step)

            # --- rollout ---
            if epoch % cfg.train.rollout_every == 0:
                reward_model.eval()
                rollout_loss = 0.0
                rollout_save_dir =  Path(self.save_dir) / "rollout"  # convert to Path first
                print("running rollout...")
                
                with torch.no_grad():
                    for num, batch in enumerate(dataloader_rollout):
                        batch = comply_lerobot_batch_regression(batch, 
                                                         camera_names=cfg.general.camera_names, 
                                                         dense_annotation=cfg.model.dense_annotation)
                        B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                        img_list = []
                        for key in self.camera_names:
                            imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                            img_list.append(imgs)
                        
                        lang_strs = batch["tasks"]
                        trg = batch["targets"].to(self.device)
                        lens = batch["lengths"].to(self.device)
                        state = batch["state"].to(self.device)
                        state = state_normalizer.normalize(state)
                        

                        # CLIP
                        # img_emb = [clip_encoder.encode_image(imgs).view(B, T, -1) for imgs in img_list]
                        # img_emb = torch.stack(img_emb, dim=1)
                        imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                        img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                        img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                        if cfg.model.dense_annotation:
                            lang_emb = torch.zeros((B, T, txt_dim), dtype=torch.float32, device=self.device)  # (B, T, txt_dim)
                            for i in range(B):
                                lang_emb[i, :, :] = clip_encoder.encode_text(lang_strs[i])
                        else:
                            lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                        if cfg.model.no_state:
                            state = torch.zeros_like(state, device=self.device)
                        reward_pred = reward_model(img_emb, lang_emb, state, lens)  # (B, T)
                        pred = torch.clip(reward_pred, 0, 1)  # (B, T)

                        length = int(lens[0].item())
                        loss = F.l1_loss(pred[0, :length], trg[0, :length], reduction="mean").item()
                        # print(f"rollout {num} / {cfg.train.rollout_steps}: loss MSE: {loss:.6f}")

                        # save results
                        result_dir = rollout_save_dir / f"epoch_{epoch:04d}"/ f"rollout_num_{num:04d}"
                        result_dir.mkdir(parents=True, exist_ok=True)
                        for i in range(T):
                            for key in self.camera_names:
                                img_save_dir = result_dir / key
                                img_save_dir.mkdir(parents=True, exist_ok=True)
                                img = batch["image_frames"][key][0, i].cpu().numpy()  # CHW, float32
                                img = np.transpose(img, (1, 2, 0))               # HWC
                                img = (img * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # Fix color channels
                                cv2.imwrite(str(img_save_dir / f"frame_{i:04d}.png"), img)
                        # save text: lang_emb, pred, and target
                        with open(result_dir / "text.txt", "w") as f:
                            f.write(f"Task: {batch['tasks'][0]}\n")
                            f.write("Stage Probabilities:\n")

                            reward_str = ", ".join(f"{x:.4f}" for x in reward_pred[0].cpu().numpy())
                            gt_str = ", ".join(f"{x:.4f}" for x in batch["targets"][0].cpu().numpy())
                            f.write(f"Reward Prediction:\n [{reward_str}]\n\n")
                            f.write(f"GT Reward:\n [{gt_str}]\n")
                            f.write(f"Single sequence mean L1 loss: {loss:.5f}\n")

                        plot_pred_vs_gt(pred[0], batch["targets"][0], batch["frame_relative_indices"][0], result_dir / "plot.png")
                        rollout_loss += loss
                        num += 1
                        if num >= cfg.train.rollout_steps:
                            break
                rollout_loss /= num
                print(f"[Rollout] Epoch {epoch} Rollout L1: {rollout_loss:.6f}")
                wandb.log({"rollout/loss": rollout_loss}, step=step)
                

            # --- clear memory ---
            del img_list, imgs_all, img_emb, lang_emb, reward_pred, pred
            torch.cuda.empty_cache()


            # --- save checkpoints ---
            save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_latest")
            
            if epoch == cfg.train.num_epochs:
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_final")
            
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(reward_model, reward_optimizer, epoch, self.save_dir, input_name="reward_best")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()

    def eval(self):
        cfg = self.cfg
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        dataset_val   = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               horizon=cfg.model.horizon, 
                                               episodes=valid_episodes, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               dense_annotation=cfg.model.dense_annotation,
                                               annotation_list=cfg.model.annotation_list)
        
        dataloader_rollout = torch.utils.data.DataLoader(dataset_val, **cfg.rollout_dataloader)
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # --- encoders ---
        # # DINO
        # vis_encoder = FrozenVisionEncoder(cfg.encoders.vision_ckpt, self.device)
        # text_encoder = FrozenTextEncoder(cfg.encoders.text_ckpt, self.device)
        # vis_dim = vis_encoder.model.config.hidden_size
        # txt_dim = text_encoder.model.config.hidden_size

        # CLIP
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_encoder = clip_encoder
        text_encoder = clip_encoder
        vis_dim = 512
        txt_dim = 512

        # reward_model_path = Path(cfg.eval.ckpt_path) / "reward_best.pt"
        reward_model_path = Path(cfg.eval.ckpt_path) / "reward_step_035000_loss_0.018.pt"

        # Create model instances
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  dense_annotation=cfg.model.dense_annotation)
        
        # Load checkpoints
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)

        # Load weights
        reward_model.load_state_dict(reward_ckpt["model"])

        # Move to device
        reward_model.to(self.device)
        reward_model.eval()

        # --- rollout ---
        rollout_loss = 0.0
        datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        rollout_save_dir =  Path(self.save_dir) / "eval" / f"{datetime_str}"  # convert to Path first
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        print("[Rollout] eval results will be saved to:", rollout_save_dir)
        
        with torch.no_grad():
            pbar = tqdm(itertools.islice(dataloader_rollout, cfg.eval.run_times), desc="Rollout", total=cfg.eval.run_times)
            for num, batch in enumerate(pbar):
                batch = comply_lerobot_batch_regression(batch, 
                                                    camera_names=cfg.general.camera_names, 
                                                    dense_annotation=cfg.model.dense_annotation)
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)
                

                # CLIP
                # img_emb = [clip_encoder.encode_image(imgs).view(B, T, -1) for imgs in img_list]
                # img_emb = torch.stack(img_emb, dim=1)
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                if cfg.model.dense_annotation:
                    lang_emb = torch.zeros((B, T, txt_dim), dtype=torch.float32, device=self.device)  # (B, T, txt_dim)
                    for i in range(B):
                        lang_emb[i, :, :] = clip_encoder.encode_text(lang_strs[i])
                else:
                    lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                reward_pred = reward_model(img_emb, lang_emb, state, lens)  # (B, T)
                pred = torch.clip(reward_pred, 0, 1)  # (B, T)

                length = int(lens[0].item())
                loss = F.l1_loss(pred[0, :length], batch["targets"][0, :length].to(self.device), reduction="mean").item()
                # print(f"rollout {num} / {cfg.train.rollout_steps}: loss MSE: {loss:.6f}")

                # save results
                result_dir = rollout_save_dir / f"rollout_num_{num:04d}"
                result_dir.mkdir(parents=True, exist_ok=True)
                for i in range(T):
                    for key in self.camera_names:
                        img_save_dir = result_dir / key
                        img_save_dir.mkdir(parents=True, exist_ok=True)
                        img = batch["image_frames"][key][0, i].cpu().numpy()  # CHW, float32
                        img = np.transpose(img, (1, 2, 0))               # HWC
                        img = (img * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # Fix color channels
                        cv2.imwrite(str(img_save_dir / f"frame_{i:04d}.png"), img)
                # save text: lang_emb, pred, and target
                with open(result_dir / "text.txt", "w") as f:
                    f.write(f"Task: {batch['tasks'][0]}\n")
                    f.write("Stage Probabilities:\n")

                    reward_str = ", ".join(f"{x:.4f}" for x in reward_pred[0].cpu().numpy())
                    pred_str = ", ".join(f"{x:.4f}" for x in pred[0].cpu().numpy())
                    gt_str = ", ".join(f"{x:.4f}" for x in batch["targets"][0].cpu().numpy())
                    f.write(f"Sub Reward Prediction:\n [{reward_str}]\n\n")
                    f.write(f"Predicted Reward:\n [{pred_str}]\n")
                    f.write(f"GT Reward:\n [{gt_str}]\n")
                    f.write(f"Single sequence mean L1 loss: {loss:.5f}\n")

                plot_pred_vs_gt(pred[0], batch["targets"][0], batch["frame_relative_indices"][0], result_dir / "plot.png")
                rollout_loss += loss
                pbar.set_postfix({"loss": loss})
                
        rollout_loss /= num
        print(f"[Rollout] Rollout L1: {rollout_loss:.6f}")

    def eval_video(self):
        import random
        cfg = self.cfg
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        dataset_val = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               horizon=cfg.model.horizon, 
                                               episodes=valid_episodes, 
                                               n_obs_steps=cfg.model.n_obs_steps, 
                                               frame_gap=cfg.model.frame_gap,
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names,
                                               dense_annotation=cfg.model.dense_annotation,
                                               annotation_list=cfg.model.annotation_list,
                                               video_eval=True)
        
        dataloader_rollout = torch.utils.data.DataLoader(dataset_val, **cfg.rollout_dataloader)
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)

        # --- encoders ---
        # # DINO
        # vis_encoder = FrozenVisionEncoder(cfg.encoders.vision_ckpt, self.device)
        # text_encoder = FrozenTextEncoder(cfg.encoders.text_ckpt, self.device)
        # vis_dim = vis_encoder.model.config.hidden_size
        # txt_dim = text_encoder.model.config.hidden_size

        # CLIP
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_encoder = clip_encoder
        text_encoder = clip_encoder
        vis_dim = 512
        txt_dim = 512

        # reward_model_path = Path(cfg.eval.ckpt_path) / "reward_best.pt"
        reward_model_path = Path(cfg.eval.ckpt_path) / "reward_step_035000_loss_0.018.pt"

        # Create model instances
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  dense_annotation=cfg.model.dense_annotation)
        

        # Load checkpoints
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        # Load weights
        reward_model.load_state_dict(reward_ckpt["model"])
        # Move to device
        reward_model.to(self.device)
        reward_model.eval()

        # save path
        datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        rollout_save_dir =  Path(self.save_dir) / "eval_video" / f"{datetime_str}"  # convert to Path first
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")
        evaled_list = []

        for i in range(cfg.eval.video_run_times):
            ep_index = random.choice([idx for idx in valid_episodes if idx not in evaled_list])
            # ep_index = valid_episodes[i]
            global_idx = valid_episodes.index(ep_index)
            evaled_list.append(ep_index)
            start_idx = dataset_val.episode_data_index["from"][global_idx].item()
            end_idx = dataset_val.episode_data_index["to"][global_idx].item()
            pred_ep_result = [0]
            gt_ep_result = [0]
            x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
            # x_offset = 0
            print(f"[Eval Video] Evaluating episode_{ep_index}, progress: {i} / {cfg.eval.video_run_times}")

            # change to use tqdm
            for idx in tqdm(range(start_idx, end_idx), desc=f"Processing episode {ep_index}"):
                data_point = dataset_val.__getitem__(idx)
                batch = comply_lerobot_batch_multi_stage_video_eval(data_point, 
                                                                    camera_names=cfg.general.camera_names, 
                                                                    dense_annotation=cfg.model.dense_annotation)
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)

                # CLIP
                # img_emb = [clip_encoder.encode_image(imgs).view(B, T, -1) for imgs in img_list]
                # img_emb = torch.stack(img_emb, dim=1)
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                if cfg.model.dense_annotation:
                    lang_emb = torch.zeros((B, T, txt_dim), dtype=torch.float32, device=self.device)  # (B, T, txt_dim)
                    for i in range(B):
                        lang_emb[i, :, :] = clip_encoder.encode_text(lang_strs[i])
                else:
                    lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                reward_pred = reward_model(img_emb, lang_emb, state, lens)  # (B, T)
                pred = torch.clip(reward_pred, 0, cfg.model.num_classes-1)  # (B, T)
                
                if abs(idx - start_idx) < (cfg.model.n_obs_steps * cfg.model.frame_gap + 100):
                    smoothed_item = pred[0, cfg.model.n_obs_steps].item()
                elif abs(idx - end_idx) < 100:
                    smoothed_item = pred[0, cfg.model.n_obs_steps].item()
                else:
                    smoothed_item = torch.mean(pred[0, 1:1+cfg.model.n_obs_steps]).item() 
                smoothed_item = min(max(smoothed_item, pred_ep_result[-1]-0.0125), pred_ep_result[-1] + 0.0125)
                pred_ep_result.append(smoothed_item)
                gt_ep_result.append(trg[0, cfg.model.n_obs_steps].item())

            # save results
            save_dir = plot_episode_result(ep_index, pred_ep_result, gt_ep_result, x_offset, rollout_save_dir)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))
            np.save(Path(save_dir) / "gt.npy", np.array(gt_ep_result))
            print(f"[Eval Video] episode_{ep_index} making video...")
            left_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/{cfg.general.repo_id}/videos/chunk-000/left_camera-images-rgb")
            middle_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/{cfg.general.repo_id}/videos/chunk-000/top_camera-images-rgb")
            right_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/{cfg.general.repo_id}/videos/chunk-000/right_camera-images-rgb")
            try:
                produce_video(rollout_save_dir, left_video_dir, middle_video_dir, right_video_dir, ep_index, x_offset)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}, progress: {i+1} / {cfg.eval.video_run_times}")


    def eval_raw_data(self):
        import random
        cfg = self.cfg
        state_normalizer = get_normalizer_from_calculated(cfg.general.state_norm_path, self.device)
        

        # CLIP
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        vis_encoder = clip_encoder
        text_encoder = clip_encoder
        vis_dim = 512
        txt_dim = 512

        reward_model_path = Path(cfg.eval.ckpt_path) 

        # Create model instances
        reward_model = RewardTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  dense_annotation=cfg.model.dense_annotation)
        

        # Load checkpoints
        reward_ckpt = torch.load(reward_model_path, map_location=self.device)
        # Load weights
        reward_model.load_state_dict(reward_ckpt["model"])
        # Move to device
        reward_model.to(self.device)
        reward_model.eval()

        # save path
        datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        rollout_save_dir =  Path(self.save_dir) / "eval_video" / f"{datetime_str}"  # convert to Path first
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")

        
        # x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
        x_offset = 18
        data_dir = cfg.eval.raw_data_dir
        run_times = cfg.eval.raw_data_run_times
        # Get all valid episode paths
        all_episodes = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("episode_")
        ]
        eval_list = all_episodes
        
        random.seed(cfg.general.seed)
        # randomly select eval_list
        if len(all_episodes) >= run_times:
            eval_list = random.sample(all_episodes, run_times)
        else:
            raise ValueError(f"Not enough episodes in {data_dir} to sample {run_times} items.")


        for i in range(run_times):
            data_path = eval_list[i]
            pred_ep_result = []
            pred_ep_smoothed = []
            # randomly select 
            ep_index = os.path.basename(data_path)
            frame_num = get_frame_num(data_path)
            traj_joint_data = get_traj_data(data_path)
            eval_frame_gap = cfg.eval.eval_frame_gap
            print(f"[EVAL_RAW]: process {i+1}/{run_times} episode: {ep_index}")
            for idx in tqdm(range(0, frame_num, eval_frame_gap), desc=f"Processing data"):
                batch = get_frame_data_fast(path=data_path, 
                                    traj_joint_data=traj_joint_data, 
                                    idx=idx,
                                    n_obs_steps=cfg.model.n_obs_steps,
                                    frame_gap=cfg.model.frame_gap,
                                    max_rewind_steps=cfg.model.max_rewind_steps,
                                    camera_names=cfg.general.camera_names,
                                    device=self.device)
                
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                    img_list.append(imgs)
                
                lang_strs = ["fold the tshirt"]
                lens = torch.tensor([1+cfg.model.n_obs_steps], dtype=torch.int32, device=self.device)
                state = batch["state"].to(self.device)
                state = state_normalizer.normalize(state)
                
                # CLIP
                # img_emb = [clip_encoder.encode_image(imgs).view(B, T, -1) for imgs in img_list]
                # img_emb = torch.stack(img_emb, dim=1)
                imgs_all = torch.cat(img_list, dim=0)  # (N * B * T, C, H, W)
                img_emb = clip_encoder.encode_image(imgs_all)  # (N * B * T, D)
                img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)  # (B, N, T, D)
                if cfg.model.dense_annotation:
                    lang_emb = torch.zeros((B, T, txt_dim), dtype=torch.float32, device=self.device)  # (B, T, txt_dim)
                    for i in range(B):
                        lang_emb[i, :, :] = clip_encoder.encode_text(lang_strs[i])
                else:
                    lang_emb = clip_encoder.encode_text(lang_strs) # lang_emb: (B, txt_dim)

                if cfg.model.no_state:
                    state = torch.zeros_like(state, device=self.device)
                reward_pred = reward_model(img_emb, lang_emb, state, lens)  # (B, T)
                pred = torch.clip(reward_pred, 0, 1)  # (B, T)
                raw_item = pred[0, cfg.model.n_obs_steps].item()
                smoothed_item = raw_item
                
                # if idx < (cfg.model.n_obs_steps * cfg.model.frame_gap + 100):
                #     smoothed_item = raw_item
                # elif abs(frame_num - idx) < 100:
                #     smoothed_item = raw_item
                # else:
                #     smoothed_item = torch.mean(pred[0, 1:1+cfg.model.n_obs_steps]).item() 
                # smoothed_item = min(max(smoothed_item, pred_ep_result[-1]-0.0125), pred_ep_result[-1] + 0.0125)
                
                pred_ep_result.append(raw_item)
                pred_ep_smoothed.append(smoothed_item)
                

            # save results
            save_dir = plot_episode_result_raw_data(ep_index, pred_ep_result, x_offset, rollout_save_dir, frame_gap=eval_frame_gap, ep_smoothed=None)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))

            print(f"[Eval Video] episode_{ep_index} making video...")
            left_video_path = Path(f"{data_path}/left_camera-images-rgb.mp4")
            middle_video_path = Path(f"{data_path}/top_camera-images-rgb.mp4")
            right_video_path = Path(f"{data_path}/right_camera-images-rgb.mp4")
            try:
                produce_video_raw_data_hybird(save_dir, left_video_path, middle_video_path, right_video_path, ep_index, cfg.model.annotation_list, x_offset, eval_frame_gap)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}")


    