from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from lerobot.common.datasets.fixed_seq_lerobot_dataset import FixedSeqLeRobotDataset
from data_utils import comply_lerobot_batch, get_valid_episodes, split_train_eval_episodes
from train_utils import set_seed, save_ckpt
from models.reward_net import RewardTransformer
from models.text_encoder import FrozenTextEncoder
from models.vision_encoder import FrozenVisionEncoder
import torch.nn as nn
import cv2
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RewindRewardWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        print(f"[Init] Using device: {self.device}")
        set_seed(cfg.general.seed)

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

        dataset_train = FixedSeqLeRobotDataset(repo_id=cfg.general.repo_id, horizon=cfg.model.horizon, episodes=train_eps, n_obs_steps=cfg.model.n_obs_steps, max_rewind_steps=cfg.model.max_rewind_steps)
        dataset_val   = FixedSeqLeRobotDataset(repo_id=cfg.general.repo_id, horizon=cfg.model.horizon, episodes=val_eps, n_obs_steps=cfg.model.n_obs_steps, max_rewind_steps=cfg.model.max_rewind_steps)

        dataloader_train = torch.utils.data.DataLoader(dataset_train, **cfg.dataloader)
        dataloader_val   = torch.utils.data.DataLoader(dataset_val, **cfg.val_dataloader)
        dataloader_rollout = torch.utils.data.DataLoader(dataset_val, **cfg.rollout_dataloader)

        # --- encoders ---
        vis_encoder = FrozenVisionEncoder(cfg.encoders.vision_ckpt, self.device)
        text_encoder = FrozenTextEncoder(cfg.encoders.text_ckpt, self.device)
        vis_dim = vis_encoder.model.config.hidden_size
        txt_dim = text_encoder.model.config.hidden_size

        # --- model ---
        model = RewardTransformer(vis_dim, txt_dim,
                                  cfg.model.n_layers,
                                  cfg.model.n_heads,
                                  cfg.model.dropout,
                                  cfg.model.max_seq_len).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr,
                                      betas=tuple(cfg.optim.betas), eps=cfg.optim.eps,
                                      weight_decay=cfg.optim.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / cfg.optim.warmup_steps,(cfg.optim.warmup_steps ** -0.5) * (s + 1) ** -0.5))

        best_val = float("inf")
        step = 0
        for epoch in range(1, cfg.train.num_epochs + 1):
            model.train()
            pbar = tqdm(dataloader_train, desc=f"Epoch {epoch}")
            for batch in pbar:
                batch = comply_lerobot_batch(batch, camera_names=cfg.general.camera_names)
                B, T = batch["image_frames"].shape[:2]
                imgs = batch["image_frames"].flatten(0, 1).to(self.device)
                lang_strs = batch["tasks"]
                trg = batch["targets"].to(self.device)
                lens = batch["lengths"].to(self.device)

                with torch.no_grad():
                    img_emb = vis_encoder(imgs).view(B, T, -1)
                    lang_emb = text_encoder(lang_strs)

                pred = model(img_emb, lang_emb, lens)
                loss = F.mse_loss(pred, trg, reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                unclipped = nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
                _ = nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                scheduler.step()

                if step % cfg.train.log_every == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": unclipped,
                        "epoch": epoch,
                        "step": step
                    })
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                step += 1

            # --- validation ---
            if epoch % cfg.train.eval_every == 0:
                model.eval()
                total_loss, num = 0.0, 0
                with torch.no_grad():
                    for batch in dataloader_val:
                        batch = comply_lerobot_batch(batch, camera_names=cfg.general.camera_names)
                        B, T = batch["image_frames"].shape[:2]
                        img_emb = vis_encoder(batch["image_frames"].flatten(0, 1).to(self.device)).view(B, T, -1)
                        lang_emb = text_encoder(batch["tasks"])
                        pred = model(img_emb, lang_emb, batch["lengths"].to(self.device))
                        total_loss += F.mse_loss(pred, batch["targets"].to(self.device), reduction="sum").item()
                        num += B * T
                val_loss = total_loss / num
                print(f"[Eval] Epoch {epoch} Val MSE: {val_loss:.6f}")
                wandb.log({"val/loss": val_loss, "epoch": epoch})

            # --- rollout ---
            if epoch % cfg.train.rollout_every == 0:
                model.eval()
                rollout_save_dir =  Path(self.save_dir) / "rollout"  # convert to Path first
                
                with torch.no_grad():
                    for num, batch in enumerate(dataloader_rollout):
                        batch = comply_lerobot_batch(batch, camera_names=cfg.general.camera_names)
                        B, T = batch["image_frames"].shape[:2]
                        img_emb = vis_encoder(batch["image_frames"].flatten(0, 1).to(self.device)).view(B, T, -1)
                        lang_emb = text_encoder(batch["tasks"])
                        pred = model(img_emb, lang_emb, batch["lengths"].to(self.device))
                        loss = F.mse_loss(pred, batch["targets"].to(self.device), reduction="sum").item()
                        print(f"rollout {num} / {cfg.train.rollout_steps}: loss MSE: {loss:.6f}")

                        # save results
                        result_dir = rollout_save_dir / f"epoch_{epoch:04d}"/ f"rollout_num_{num:04d}"
                        result_dir.mkdir(parents=True, exist_ok=True)
                        for i in range(T):
                            img = batch["image_frames"][0, i].cpu().numpy()  # CHW, float32
                            img = np.transpose(img, (1, 2, 0))               # HWC
                            img = (img * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # Fix color channels
                            cv2.imwrite(str(result_dir / f"frame_{i:04d}.png"), img)
                        # save text: lang_emb, pred, and target
                        with open(result_dir / "text.txt", "w") as f:
                            f.write(f"Task: {batch['tasks'][0]}\n")
                            f.write(f"Predicted Reward: {pred[0].cpu().numpy()}\n")
                            f.write(f"GT Reward: {batch['targets'][0].cpu().numpy()}\n")
                            f.write(f"single sequence loss: {loss:.6f}\n")

                        num += 1
                        if num >= cfg.train.rollout_steps:
                            break

            # --- save checkpoints ---
            if epoch % cfg.train.save_every == 0:
                save_ckpt(model, optimizer, epoch, self.save_dir)
            elif epoch == cfg.train.num_epochs:
                save_ckpt(model, optimizer, epoch, self.save_dir, input_name="final")
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(model, optimizer, epoch, self.save_dir, input_name="min_loss")

        print(f"Training done. Best val_loss MSE = {best_val}")
        wandb.finish()
