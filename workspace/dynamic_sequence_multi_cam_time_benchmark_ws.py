from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from lerobot.common.datasets.dynamic_seq_lerobot_dataset import DynamicSeqLeRobotDataset 
from data_utils import comply_lerobot_batch, get_valid_episodes, split_train_eval_episodes
from train_utils import set_seed, save_ckpt
from models.reward_net import RewardTransformer
from models.text_encoder import FrozenTextEncoder
from models.vision_encoder import FrozenVisionEncoder
from models.clip_encoder import FrozenCLIPEncoder
import torch.nn as nn
import cv2
import numpy as np
import time

import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"



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
        # --- wandb ---
        wandb.init(
            project=f'{cfg.general.project_name}-{cfg.general.task_name}',
            name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
            config=cfg,
        )

        # --- data ---
        valid_episodes = get_valid_episodes(cfg.general.repo_id)
        train_eps, val_eps = split_train_eval_episodes(valid_episodes, 1 - cfg.train.val_portion, seed=cfg.general.seed)

        dataset_train = DynamicSeqLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               episodes=train_eps, 
                                               n_seq=cfg.model.n_seq, 
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names)
        
        dataset_val   = DynamicSeqLeRobotDataset(repo_id=cfg.general.repo_id, 
                                               episodes=val_eps, 
                                               n_seq=cfg.model.n_seq, 
                                               max_rewind_steps=cfg.model.max_rewind_steps,
                                               image_names=cfg.general.camera_names)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg.dataloader)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val, **cfg.val_dataloader)
        dataloader_rollout = torch.utils.data.DataLoader(dataset_val, **cfg.rollout_dataloader)

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
        reward_model = RewardTransformer(cfg.model.d_model, 
                                  vis_dim, txt_dim,
                                  cfg.model.n_layers,
                                  cfg.model.n_heads,
                                  cfg.model.dropout,
                                  cfg.model.max_seq_len,
                                  len(self.camera_names)).to(self.device)

        optimizer = torch.optim.AdamW(reward_model.parameters(), lr=cfg.optim.lr,
                                      betas=tuple(cfg.optim.betas), eps=cfg.optim.eps,
                                      weight_decay=cfg.optim.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / cfg.optim.warmup_steps,(cfg.optim.warmup_steps ** -0.5) * (s + 1) ** -0.5))

        best_val = float("inf")
        step = 0

        
        for epoch in range(1, cfg.train.num_epochs + 1):
            data_loading_start = time.perf_counter()
            data_loading_flag = False
            reward_model.train()
            pbar = tqdm(dataloader_train, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                start_time = time.perf_counter()
                if not data_loading_flag:
                    print(f"data loading time {(start_time - data_loading_start):.4f}s")
                    data_loading_flag = True

                # ================== Data I/O ==================
                io_start = time.perf_counter()
                batch = comply_lerobot_batch(batch, camera_names=cfg.general.camera_names)
                B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                img_list = []
                for key in self.camera_names:
                    imgs = batch["image_frames"][key].flatten(0, 1).to(self.device)  # (B*T, C, H, W)
                    img_list.append(imgs)
                    # print(imgs.dtype)
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)
                io_time = time.perf_counter() - io_start

                # ================== Embedding ==================
                emb_start = time.perf_counter()
                with torch.no_grad():
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
                    lang_emb = clip_encoder.encode_text(lang_strs)
                
                emb_time = time.perf_counter() - emb_start

                # ================== Forward ==================
                fwd_start = time.perf_counter()
                pred = reward_model(img_emb, lang_emb, lens)
                loss = F.mse_loss(pred, trg, reduction="mean")
                fwd_time = time.perf_counter() - fwd_start

                # ================== Backward ==================
                bwd_start = time.perf_counter()
                optimizer.zero_grad()
                loss.backward()
                unclipped = nn.utils.clip_grad_norm_(reward_model.parameters(), float("inf")).item()
                _ = nn.utils.clip_grad_norm_(reward_model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                scheduler.step()
                bwd_time = time.perf_counter() - bwd_start

                # ================== Logging ==================
                log_time = time.perf_counter() - start_time

                if step % cfg.train.log_every == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": unclipped,
                        "epoch": epoch,
                    }, step=step)

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                print(
                    f"data={io_time:.4f}s, embed={emb_time:.4f}s, "
                    f"fwd={fwd_time:.4f}s, bwd={bwd_time:.4f}s, total={log_time:.4f}s"
                )

                step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                reward_model.eval()
                total_loss, num = 0.0, 0
                print("running validation...")
                with torch.no_grad():
                    for batch in dataloader_val:
                        batch = comply_lerobot_batch(batch, camera_names=cfg.general.camera_names)
                        B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                        img_list = []
                        for key in self.camera_names:
                            imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                            img_list.append(imgs)
                        
                        lang_strs = batch["tasks"]
                        trg = batch["targets"].to(self.device)
                        lens = batch["lengths"].to(self.device)

                        with torch.no_grad():
                            img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]
                            img_emb = torch.stack(img_emb, dim=1)  
                            lang_emb = text_encoder(lang_strs)
                        # img_emb: (B, N, T, vis_dim), lang_emb: (B, txt_dim)
                        pred = reward_model(img_emb, lang_emb, lens)
                        total_loss += F.l1_loss(pred, batch["targets"].to(self.device), reduction="mean").item()
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
                        batch = comply_lerobot_batch(batch, camera_names=cfg.general.camera_names)
                        B, T = batch["image_frames"][self.camera_names[0]].shape[:2]
                        img_list = []
                        for key in self.camera_names:
                            imgs = batch["image_frames"][key].flatten(0, 1).to(self.device) # (B*T, C, H, W)
                            img_list.append(imgs)
                        
                        lang_strs = batch["tasks"]
                        trg = batch["targets"].to(self.device)
                        lens = batch["lengths"].to(self.device)

                        with torch.no_grad():
                            img_emb = [vis_encoder(imgs).view(B, T, -1) for imgs in img_list]
                            img_emb = torch.stack(img_emb, dim=1)  
                            lang_emb = text_encoder(lang_strs)
                        # img_emb: (B, N, T, vis_dim), lang_emb: (B, txt_dim)
                        pred = reward_model(img_emb, lang_emb, lens)
                        loss = F.l1_loss(pred, batch["targets"].to(self.device), reduction="mean").item()
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
                            pred_str = ", ".join(f"{x:.4f}" for x in pred[0].cpu().numpy())
                            gt_str = ", ".join(f"{x:.4f}" for x in batch["targets"][0].cpu().numpy())
                            f.write(f"Predicted Reward:\n [{pred_str}]\n")
                            f.write(f"GT Reward:\n [{gt_str}]\n")
                            f.write(f"Single sequence mean L1 loss: {loss:.5f}\n")

                        rollout_loss += loss
                        num += 1
                        if num >= cfg.train.rollout_steps:
                            break
                rollout_loss /= num
                print(f"[Rollout] Epoch {epoch} Rollout L1: {rollout_loss:.6f}")
                wandb.log({"rollout/loss": rollout_loss}, step=step)

            # --- save checkpoints ---
            save_ckpt(reward_model, optimizer, epoch, self.save_dir, input_name="latest")
            
            if epoch == cfg.train.num_epochs:
                save_ckpt(reward_model, optimizer, epoch, self.save_dir, input_name="final")
            elif epoch % cfg.train.save_every == 0:
                save_ckpt(reward_model, optimizer, epoch, self.save_dir)
            
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(reward_model, optimizer, epoch, self.save_dir, input_name="best")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()
