import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path


def draw_plot_frame(step: int, pred, gt, x_offset, width=448, height=448):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

    timesteps = np.arange(len(pred)) + x_offset
    ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    ax.plot(timesteps, gt, label='Ground Truth', linewidth=2)
    ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

     # === Add Milestone Text ===
    current_gt = gt[step]
    text_y = ax.get_ylim()[1] * 0.9  # position near top

    if 1.0 <= current_gt < 2.0:
        ax.text(step + x_offset, text_y, "Grasp & Pick Up", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif current_gt <= 1.0:
        ax.text(step + x_offset, text_y, "Reach Cube", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    else:
        ax.text(step + x_offset, text_y, "Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')

    # canvas = FigureCanvas(fig)
    # canvas.draw()
    # img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)


    return img



def produce_video(save_dir, left_video_dir, middle_video_dir, episode_num, x_offset=30):
    # === CONFIGURATION ===
    episode_dir = save_dir / f"episode_{episode_num}"
    left_video_path = left_video_dir / f"episode_{episode_num:06d}.mp4"
    middle_video_path = middle_video_dir / f"episode_{episode_num:06d}.mp4"
    pred_path = episode_dir / "pred.npy"
    pred_path = episode_dir / "pred.npy"
    gt_path = episode_dir / "gt.npy"
    output_path = episode_dir / "combined_video.mp4"
    frame_rate = 30

    target_h, target_w = 448, 448  # resolution per panel

    # === LOAD DATA ===
    pred_full = np.load(pred_path)
    gt_full = np.load(gt_path)
    pred = pred_full[x_offset:]
    gt = gt_full[x_offset:]
    T = len(pred)

    # Load videos
    clip_left = VideoFileClip(str(left_video_path))
    clip_middle = VideoFileClip(str(middle_video_path))
    frames_left = [f for f in clip_left.iter_frames(fps=frame_rate)][x_offset:x_offset + T]
    frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset:x_offset + T]

    assert len(frames_left) >= T and len(frames_middle) >= T, "Video(s) too short"

    # === CREATE COMBINED FRAMES ===
    combined_frames = []
    for t in range(T):
        left_resized = cv2.resize(frames_left[t], (target_w, target_h))
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame(t, pred, gt, x_offset, height=target_h, width=target_w)

        combined = np.concatenate((left_resized, middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # === SAVE VIDEO ===
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec='libx264')

def main():
    episode_num = 257
    x_offset = 30
    episode_dir = Path(f"/home/david_chen/rewind_reproduce/outputs/2025-07-16/12-18-19/rewind_reward_fixed_seq_model_frame_gap_multi_stage/pick_up_cube/eval_video/2025.07.16-12.18.26")
    left_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/Qianzhong-Chen/yam_pick_up_cube_sim_rotate_reward_two_stage_0715/videos/chunk-000/top_camera-images-rgb")
    middle_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/Qianzhong-Chen/yam_pick_up_cube_sim_rotate_reward_two_stage_0715/videos/chunk-000/right_camera-images-rgb")
    produce_video(episode_dir, left_video_dir, middle_video_dir, episode_num, x_offset)

if __name__ == "__main__":
    main()
