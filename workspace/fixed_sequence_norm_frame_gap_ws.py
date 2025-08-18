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
from data_utils import comply_lerobot_batch_norm, get_valid_episodes, split_train_eval_episodes, comply_lerobot_batch_multi_stage_video_eval
from train_utils import plot_episode_result, set_seed, save_ckpt, plot_pred_vs_gt, get_normalizer_from_calculated, plot_episode_result, plot_episode_result_raw_data
from raw_data_utils import get_frame_num, get_frame_data, get_traj_data
from models.multi_stage_reward_net import RewardTransformer
from models.multi_stage_estimate_net import StageTransformer
from models.text_encoder import FrozenTextEncoder
from models.vision_encoder import FrozenVisionEncoder
from models.clip_encoder import FrozenCLIPEncoder
from make_demo_video import produce_video, produce_video_raw_data
import torch.nn as nn
import cv2
import numpy as np
import time
from torchvision.utils import save_image
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"] = "**/rollout/**"
# os.environ["WANDB_MODE"] = "disabled"


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

        dataset_val   = FrameGapLeRobotDataset(repo_id=cfg.general.repo_id, 
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
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  num_classes=cfg.model.num_classes,
                                  dense_annotation=cfg.model.dense_annotation).to(self.device)


        # Optimizer
        stage_optimizer = torch.optim.AdamW(
            stage_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )


        # Schedulers

        # Stage scheduler
        stage_warmup_scheduler = LinearLR(
            stage_optimizer,
            start_factor=1e-6 / cfg.optim.lr,  # can be 0.0 if you prefer
            end_factor=1.0,
            total_iters=cfg.optim.warmup_steps
        )
        stage_cosine_scheduler = CosineAnnealingLR(
            stage_optimizer,
            T_max=cfg.optim.total_steps - cfg.optim.warmup_steps,
            eta_min=0.0
        )
        stage_scheduler = SequentialLR(
            stage_optimizer,
            schedulers=[stage_warmup_scheduler, stage_cosine_scheduler],
            milestones=[cfg.optim.warmup_steps]
        )


        best_val = float("inf")
        step = 0
        for epoch in range(1, cfg.train.num_epochs + 1):
            stage_model.train()
            with tqdm(dataloader_train, desc=f"Epoch {epoch}") as pbar:
                for batch in pbar:
                    batch = comply_lerobot_batch_norm(batch, 
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
                    gt_stage = torch.clamp((trg * cfg.model.num_classes).round().long(), max=cfg.model.num_classes - 1)

                    with torch.no_grad():
                        state = state_normalizer.normalize(state)
                        # # DINO
                        # # img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]  
                        # # img_emb = torch.stack(img_emb, dim=1)
                        # imgs_all = torch.cat(img_list, dim=0)  # list of tensors (B*T, C, H, W) → (N*B*T, C, H, W)
                        # img_emb = vis_encoder(imgs_all)
                        # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)
                        # lang_emb = text_encoder(lang_strs)

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
                    stage_pred = stage_model(img_emb, lang_emb, state, lens)  # (B, N, T, num_classes)
                    stage_loss = F.cross_entropy(stage_pred.view(-1, cfg.model.num_classes), gt_stage.view(-1), reduction="mean")
                    # TODO: debug possible issue with "mask", check network structure, is sequence really works?

                    stage_optimizer.zero_grad()
                    stage_loss.backward()
                    stage_unclipped = nn.utils.clip_grad_norm_(stage_model.parameters(), float("inf")).item()
                    _ = nn.utils.clip_grad_norm_(stage_model.parameters(), cfg.train.grad_clip)
                    stage_optimizer.step()
                    stage_scheduler.step()

                    if step % cfg.train.log_every == 0:
                        wandb.log({
                            "train/stage_loss": stage_loss.item(),
                            "train/stage_grad_norm": stage_unclipped,
                            "epoch": epoch,
                        }, step=step)
                    
                    pbar.set_postfix(loss=f"{(stage_loss.item()):.4f}")

                    if step % cfg.train.save_every == 0:
                        save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name=f"stage_step_{step:06d}_loss_{stage_loss.item():.3f}")

                    step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                stage_model.eval()
                total_loss, num = 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    for batch in dataloader_val:
                        batch = comply_lerobot_batch_norm(batch, 
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

                        gt_stage = torch.clamp((trg * cfg.model.num_classes).round().long(), max=cfg.model.num_classes - 1)


                        # # DINO
                        # # img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]  
                        # # img_emb = torch.stack(img_emb, dim=1)
                        # imgs_all = torch.cat(img_list, dim=0)  # list of tensors (B*T, C, H, W) → (N*B*T, C, H, W)
                        # img_emb = vis_encoder(imgs_all)
                        # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)
                        # lang_emb = text_encoder(lang_strs)

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
                        
                        stage_pred = stage_model(img_emb, lang_emb, state, lens)  # (B, T, num_classes)

                        stage_loss = F.cross_entropy(stage_pred.view(-1, cfg.model.num_classes), gt_stage.view(-1), reduction="mean")
                        total_loss += stage_loss.item()
                        num += 1

                val_loss = total_loss / num 
                print(f"[Eval] Epoch {epoch} Val L1: {val_loss:.6f}")
                wandb.log({"val/loss": val_loss}, step=step)

            # --- rollout ---
            if epoch % cfg.train.rollout_every == 0:
                stage_model.eval()
                rollout_loss = 0.0
                rollout_save_dir =  Path(self.save_dir) / "rollout"  # convert to Path first
                print("running rollout...")
                
                with torch.no_grad():
                    for num, batch in enumerate(dataloader_rollout):
                        batch = comply_lerobot_batch_norm(batch, 
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
                        

                        # # DINO
                        # # img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]  
                        # # img_emb = torch.stack(img_emb, dim=1)
                        # imgs_all = torch.cat(img_list, dim=0)  # list of tensors (B*T, C, H, W) → (N*B*T, C, H, W)
                        # img_emb = vis_encoder(imgs_all)
                        # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)
                        # lang_emb = text_encoder(lang_strs)

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
                        stage_prob = stage_model(img_emb, lang_emb, state, lens).softmax(dim=-1)  # (B, T, num_classes)
                        stage_pred = stage_prob.argmax(dim=-1)  # (B, T)
                        pred = stage_pred.float() / (cfg.model.num_classes - 1)

                        length = int(lens[0].item())
                        loss = F.l1_loss(pred[0, :length], batch["targets"][0, :length].to(self.device), reduction="mean").item()
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
                            stage_prob_np = stage_prob[0].cpu().numpy().T  # (num_classes, T)
                            f.write("Stage Probabilities:\n")
                            for class_idx, row in enumerate(stage_prob_np):
                                row_str = ", ".join(f"{p:.4f}" for p in row)
                                f.write(f"Class {class_idx}: [{row_str}]\n\n")

                            pred_str = ", ".join(f"{x:.4f}" for x in pred[0].cpu().numpy())
                            gt_str = ", ".join(f"{x:.4f}" for x in batch["targets"][0].cpu().numpy())
                            f.write(f"Predicted Reward:\n [{pred_str}]\n")
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
            del img_list, imgs_all, img_emb, lang_emb, stage_pred, pred, stage_prob, stage_prob_np
            torch.cuda.empty_cache()


            # --- save checkpoints ---
            save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name="stage_latest")
            
            if epoch == cfg.train.num_epochs:
                save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name="stage_final")
            
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, input_name="stage_best")

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

        # stage_model_path = Path(cfg.eval.ckpt_path) / "stage_best.pt"
        stage_model_path = Path(cfg.eval.ckpt_path) / "stage_step_035000_loss_0.089.pt"

        # Create model instances
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  num_classes=cfg.model.num_classes,
                                  dense_annotation=cfg.model.dense_annotation)

        # Load checkpoints
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        # Load weights
        stage_model.load_state_dict(stage_ckpt["model"])
        # Move to device
        stage_model.to(self.device)
        stage_model.eval()

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
                batch = comply_lerobot_batch_norm(batch, 
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
                

                # # DINO
                # # img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]  
                # # img_emb = torch.stack(img_emb, dim=1)
                # imgs_all = torch.cat(img_list, dim=0)  # list of tensors (B*T, C, H, W) → (N*B*T, C, H, W)
                # img_emb = vis_encoder(imgs_all)
                # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)
                # lang_emb = text_encoder(lang_strs)

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
                stage_prob = stage_model(img_emb, lang_emb, state, lens).softmax(dim=-1)  # (B, T, num_classes)
                stage_pred = stage_prob.argmax(dim=-1)  # (B, T)
                pred = stage_pred.float() / (cfg.model.num_classes - 1)  # (B, T) 

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
                    stage_prob_np = stage_prob[0].cpu().numpy().T  # (num_classes, T)
                    f.write("Stage Probabilities:\n")
                    for class_idx, row in enumerate(stage_prob_np):
                        row_str = ", ".join(f"{p:.4f}" for p in row)
                        f.write(f"Class {class_idx}: [{row_str}]\n\n")

                    stage_str = ", ".join(f"{x:.4f}" for x in stage_pred[0].cpu().numpy())
                    pred_str = ", ".join(f"{x:.4f}" for x in pred[0].cpu().numpy())
                    gt_str = ", ".join(f"{x:.4f}" for x in batch["targets"][0].cpu().numpy())
                    f.write(f"Stage Prediction:\n [{stage_str}]\n")
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

        # stage_model_path = Path(cfg.eval.ckpt_path) / "stage_best.pt"
        stage_model_path = Path(cfg.eval.ckpt_path) / "stage_step_035000_loss_0.089.pt"

        # Create model instances
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  num_classes=cfg.model.num_classes,
                                  dense_annotation=cfg.model.dense_annotation)

        # Load checkpoints
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        # Load weights
        stage_model.load_state_dict(stage_ckpt["model"])
        # Move to device
        stage_model.to(self.device)
        stage_model.eval()

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
                
                # # DINO
                # # img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]  
                # # img_emb = torch.stack(img_emb, dim=1)
                # imgs_all = torch.cat(img_list, dim=0)  # list of tensors (B*T, C, H, W) → (N*B*T, C, H, W)
                # img_emb = vis_encoder(imgs_all)
                # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)
                # lang_emb = text_encoder(lang_strs)

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
                stage_prob = stage_model(img_emb, lang_emb, state, lens).softmax(dim=-1)  # (B, T, num_classes)
                stage_pred = stage_prob.argmax(dim=-1)  # (B, T)
                pred = stage_pred.float() / (cfg.model.num_classes - 1)  # (B, T)

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

        # stage_model_path = Path(cfg.eval.ckpt_path) / "stage_best.pt"
        stage_model_path = Path(cfg.eval.ckpt_path) / "stage_step_035000_loss_0.089.pt"

        # Create model instances
        stage_model = StageTransformer(d_model=cfg.model.d_model, 
                                  vis_emb_dim=vis_dim, 
                                  text_emb_dim=txt_dim,
                                  state_dim=cfg.model.state_dim,
                                  n_layers=cfg.model.n_layers,
                                  n_heads=cfg.model.n_heads,
                                  dropout=cfg.model.dropout,
                                  max_seq_len=cfg.model.max_seq_len,
                                  num_cameras=len(self.camera_names),
                                  num_classes=cfg.model.num_classes,
                                  dense_annotation=cfg.model.dense_annotation)

        # Load checkpoints
        stage_ckpt = torch.load(stage_model_path, map_location=self.device)
        # Load weights
        stage_model.load_state_dict(stage_ckpt["model"])
        # Move to device
        stage_model.to(self.device)
        stage_model.eval()

        # save path
        datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        rollout_save_dir =  Path(self.save_dir) / "eval_video" / f"{datetime_str}"  # convert to Path first
        rollout_save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, rollout_save_dir / "config.yaml")

        
        x_offset = cfg.model.frame_gap * cfg.model.n_obs_steps
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
            pred_ep_result = [0]
            # randomly select 
            ep_index = os.path.basename(data_path)
            frame_num = get_frame_num(data_path)
            traj_joint_data = get_traj_data(data_path)
            print(f"[EVAL_RAW]: process {i+1}/{run_times} episode: {ep_index}")
            for idx in tqdm(range(frame_num), desc=f"Processing data"):
                batch = get_frame_data(path=data_path, 
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
                
                # # DINO
                # # img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]  
                # # img_emb = torch.stack(img_emb, dim=1)
                # imgs_all = torch.cat(img_list, dim=0)  # list of tensors (B*T, C, H, W) → (N*B*T, C, H, W)
                # img_emb = vis_encoder(imgs_all)
                # img_emb = img_emb.view(len(img_list), B, T, -1).permute(1, 0, 2, 3)
                # lang_emb = text_encoder(lang_strs)

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
                stage_prob = stage_model(img_emb, lang_emb, state, lens).softmax(dim=-1)  # (B, T, num_classes)
                stage_pred = stage_prob.argmax(dim=-1)  # (B, T)
                pred = stage_pred.float() / (cfg.model.num_classes - 1)  # (B, T)                
                if idx < (cfg.model.n_obs_steps * cfg.model.frame_gap + 100):
                    smoothed_item = pred[0, cfg.model.n_obs_steps].item()
                elif abs(frame_num - idx) < 100:
                    smoothed_item = pred[0, cfg.model.n_obs_steps].item()
                else:
                    smoothed_item = torch.mean(pred[0, 1:1+cfg.model.n_obs_steps]).item() 
                smoothed_item = min(max(smoothed_item, pred_ep_result[-1]-0.0125), pred_ep_result[-1] + 0.0125)
                
                # if idx < (cfg.model.n_obs_steps * cfg.model.frame_gap + 100):
                #     smoothed_item = pred[0, cfg.model.n_obs_steps].item()
                # elif abs(frame_num - idx) < 100:
                #     smoothed_item = pred[0, cfg.model.n_obs_steps].item()
                # else:
                #     smoothed_item = torch.mean(pred[0, 5:1+cfg.model.n_obs_steps]).item() 
                # smoothed_item = min(max(smoothed_item, pred_ep_result[-1]-0.02), pred_ep_result[-1] + 0.02)
                
                pred_ep_result.append(smoothed_item)
                

            # save results
            save_dir = plot_episode_result_raw_data(ep_index, pred_ep_result, x_offset, rollout_save_dir)
            np.save(Path(save_dir) / "pred.npy", np.array(pred_ep_result))

            print(f"[Eval Video] episode_{ep_index} making video...")
            left_video_path = Path(f"{data_path}/left_camera-images-rgb.mp4")
            middle_video_path = Path(f"{data_path}/top_camera-images-rgb.mp4")
            right_video_path = Path(f"{data_path}/right_camera-images-rgb.mp4")
            try:
                produce_video_raw_data(save_dir, left_video_path, middle_video_path, right_video_path, ep_index, x_offset)
            except Exception as e:
                print(f"[Eval Video] episode_{ep_index} video production failed: {e}")
            
            print(f"[Eval Video] episode_{ep_index} results saved to: {save_dir}")


   