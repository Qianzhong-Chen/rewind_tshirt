import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path
import os


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

    # Sparse anno
    # annotation_list= ["Grab the tshirt from the pile", 
    #                   "Move the tshirt to the center of the board",
    #                   "Flatten the tshirt out",
    #                   "Fold the tshirt",
    #                   "Neatly place the folded tshirt to the corner",
    #                   "task finished"]
    # if current_gt <= 1.0:
    #     ax.text(step + x_offset, text_y, "Grab the tshirt from the pile", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 1.0 <= current_gt < 2.0:
    #     ax.text(step + x_offset, text_y, "Move the tshirt to the center of the board", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 2.0 <= current_gt < 3.0:
    #     ax.text(step + x_offset, text_y, "Flatten the tshirt out", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 3.0 <= current_gt < 4.0:
    #     ax.text(step + x_offset, text_y, "Fold the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 4.0 <= current_gt < 5.0:
    #     ax.text(step + x_offset, text_y, "Neatly place the folded tshirt to the corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # else:
    #     ax.text(step + x_offset, text_y, "Task Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    
    # Dense anno
    annotation_list= [
            "grab crumpled tshirt and move to center",
            "flatten out the tshirt",
            "grab near side and fold one-third",
            "grab far side and fold into rectangle",
            "rotate the tshirt 90 degrees",
            "grab bottom and fold one-third",
            "grab two-third side and fold into square",
            "put folded tshirt into corner"
        ]
    if current_gt <= 1.0:
        ax.text(step + x_offset, text_y, "grab crumpled tshirt and move to center", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 1.0 <= current_gt < 2.0:
        ax.text(step + x_offset, text_y, "flatten out the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 2.0 <= current_gt < 3.0:
        ax.text(step + x_offset, text_y, "grab near side and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 3.0 <= current_gt < 4.0:
        ax.text(step + x_offset, text_y, "grab far side and fold into rectangle", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 4.0 <= current_gt < 5.0:
        ax.text(step + x_offset, text_y, "rotate the tshirt 90 degrees", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 5.0 <= current_gt < 6.0:
        ax.text(step + x_offset, text_y, "grab bottom and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 6.0 <= current_gt < 7.0:
        ax.text(step + x_offset, text_y, "grab two-third side and fold into square", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 7.0 <= current_gt < 8.0:
        ax.text(step + x_offset, text_y, "put folded tshirt into corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    else:
        ax.text(step + x_offset, text_y, "Task Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')


    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)

    return img

def draw_plot_frame_raw_data(step: int, pred, x_offset, width=448, height=448):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

    timesteps = np.arange(len(pred)) + x_offset
    ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

     # === Add Milestone Text ===
    current_pred = pred[step]
    text_y = ax.get_ylim()[1] * 0.9  # position near top

    # Sparse anno
    # if current_pred <= 1.0:
    #     ax.text(step + x_offset, text_y, "Grab the tshirt from the pile", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 1.0 <= current_pred < 2.0:
    #     ax.text(step + x_offset, text_y, "Move the tshirt to the center of the board", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 2.0 <= current_pred < 3.0:
    #     ax.text(step + x_offset, text_y, "Flatten the tshirt out", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 3.0 <= current_pred < 4.0:
    #     ax.text(step + x_offset, text_y, "Fold the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # elif 4.0 <= current_pred < 5.0:
    #     ax.text(step + x_offset, text_y, "Neatly place the folded tshirt to the corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    # else:
    #     ax.text(step + x_offset, text_y, "Task Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    
    # Dense anno
    annotation_list= [
            "grab crumpled tshirt and move to center",
            "flatten out the tshirt",
            "grab near side and fold one-third",
            "grab far side and fold into rectangle",
            "rotate the tshirt 90 degrees",
            "grab bottom and fold one-third",
            "grab two-third side and fold into square",
            "put folded tshirt into corner"
        ]
    if current_pred <= 1.0:
        ax.text(step + x_offset, text_y, "grab crumpled tshirt and move to center", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 1.0 <= current_pred < 2.0:
        ax.text(step + x_offset, text_y, "flatten out the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 2.0 <= current_pred < 3.0:
        ax.text(step + x_offset, text_y, "grab near side and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 3.0 <= current_pred < 4.0:
        ax.text(step + x_offset, text_y, "grab far side and fold into rectangle", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 4.0 <= current_pred < 5.0:
        ax.text(step + x_offset, text_y, "rotate the tshirt 90 degrees", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 5.0 <= current_pred < 6.0:
        ax.text(step + x_offset, text_y, "grab bottom and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 6.0 <= current_pred < 7.0:
        ax.text(step + x_offset, text_y, "grab two-third side and fold into square", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    elif 7.0 <= current_pred < 8.0:
        ax.text(step + x_offset, text_y, "put folded tshirt into corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    else:
        ax.text(step + x_offset, text_y, "Task Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')


    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)

    return img

def draw_plot_frame_raw_data_hybird(step: int, pred, x_offset, annotation_list, width=448, height=448):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

    timesteps = np.arange(len(pred)) + x_offset
    ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # === Add Milestone Text ===
    current_pred = float(pred[step])
    text_y = ax.get_ylim()[1] * 0.9  # position near top

    # Map prediction -> stage index (0..len-1). Out-of-range -> finished.
    stage_idx = int(np.floor(current_pred))
    if stage_idx < 0:
        stage_idx = 0  # clamp negatives to first stage

    if stage_idx >= len(annotation_list):
        label_text = "Task Finished"
    else:
        label_text = annotation_list[stage_idx]

    ax.text(step + x_offset, text_y, label_text,
            color='green', fontsize=12, fontweight='bold', ha='center', va='top')

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)

    return img

def draw_plot_frame_raw_data_norm(step: int, pred, x_offset, width=448, height=448, frame_gap=None, conf=None, smoothed=None):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

    # === Timesteps ===
    if frame_gap is None:
        timesteps = np.arange(len(pred)) + x_offset
    else:
        timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset

    # === Plot raw prediction ===
    line_pred, = ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    handles, labels = [line_pred], ["Predicted"]

    # === Plot smoothed prediction ===
    if smoothed is not None:
        line_smooth, = ax.plot(timesteps, smoothed, label="Smoothed", linewidth=2, color="orange")
        handles.append(line_smooth)
        labels.append("Smoothed")

    # === Vertical line at current step ===
    if frame_gap is None:
        ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    else:
        ax.axvline(x=step * frame_gap + x_offset, color='r', linestyle='--', linewidth=2)

    # === Labels and style ===
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # === Confidence curve on twin axis ===
    if conf is not None:
        ax2 = ax.twinx()
        line_conf, = ax2.plot(timesteps, conf, linestyle=':', color='green', label="Confidence")
        ax2.set_ylabel("Confidence")
        handles.append(line_conf)
        labels.append("Confidence")

    # === Legend ===
    ax.legend(handles, labels, loc="best")

    fig.tight_layout()

    # === Render to image ===
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # RGB only
    plt.close(fig)

    return img

def produce_video(save_dir, left_video_dir, middle_video_dir, right_video_dir, episode_num, x_offset=30):
    # === CONFIGURATION ===
    episode_dir = save_dir / f"episode_{episode_num}"
    left_video_path = left_video_dir / f"episode_{episode_num:06d}.mp4"
    middle_video_path = middle_video_dir / f"episode_{episode_num:06d}.mp4"
    right_video_path = right_video_dir / f"episode_{episode_num:06d}.mp4"
    pred_path = episode_dir / "pred.npy"
    gt_path = episode_dir / "gt.npy"
    output_path = episode_dir / "combined_video.mp4"
    frame_rate = 32

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
    clip_right = VideoFileClip(str(right_video_path))
    frames_left = [f for f in clip_left.iter_frames(fps=frame_rate)][x_offset:]
    frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset:]
    frames_right = [f for f in clip_right.iter_frames(fps=frame_rate)][x_offset:]
    assert len(frames_left) >= T and len(frames_middle) >= T and len(frames_right) >= T, "Video(s) too short"
    total_len = len(frames_left)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_left = [frames_left[i] for i in indices]
    frames_right = [frames_right[i] for i in indices]
    frames_middle = [frames_middle[i] for i in indices]


    # === CREATE COMBINED FRAMES ===
    combined_frames = []
    for t in range(T):
        left_resized = cv2.resize(frames_left[t], (target_w, target_h))
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        right_resized = cv2.resize(frames_right[t], (target_w, target_h))
        plot_img = draw_plot_frame(t, pred, gt, x_offset, height=target_h, width=target_w)

        combined = np.concatenate((left_resized, middle_resized, right_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # === SAVE VIDEO ===
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec='libx264')

def produce_video_raw_data(save_dir, left_video_path, middle_video_path, right_video_path, episode_num, x_offset=30):
    # === CONFIGURATION ===
    save_dir = Path(save_dir)
    pred_path = save_dir / "pred.npy"
    output_path = save_dir / "combined_video.mp4"
    frame_rate = 35

    target_h, target_w = 448, 448  # resolution per panel

    # === LOAD DATA ===
    pred_full = np.load(pred_path)
    pred = pred_full[x_offset:]
    T = len(pred)

    # Load videos
    clip_middle = VideoFileClip(str(middle_video_path))
    
    frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset:]
    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(f"WARNING: Not enough frames in videos. Expected {T}, found {min_frames_num}. Adjusting to available frames.")
        T = min_frames_num
        pred = pred[gap:]
    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_middle = [frames_middle[i] for i in indices]


    # === CREATE COMBINED FRAMES ===
    combined_frames = []
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame_raw_data(t, pred, x_offset, height=target_h, width=target_w)

        combined = np.concatenate((middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # === SAVE VIDEO ===
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec='libx264')
    
def produce_video_raw_data_hybird(save_dir, left_video_path, middle_video_path, right_video_path, episode_num, annotation_list, x_offset=30):
    # === CONFIGURATION ===
    save_dir = Path(save_dir)
    pred_path = save_dir / "pred.npy"
    output_path = save_dir / "combined_video.mp4"
    frame_rate = 32

    target_h, target_w = 448, 448  # resolution per panel

    # === LOAD DATA ===
    pred_full = np.load(pred_path)
    pred = pred_full[x_offset:]
    T = len(pred)

    # Load videos
    clip_middle = VideoFileClip(str(middle_video_path))
    frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset:]
    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(f"WARNING: Not enough frames in videos. Expected {T}, found {min_frames_num}. Adjusting to available frames.")
        T = min_frames_num
        pred = pred[gap:]
    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_middle = [frames_middle[i] for i in indices]


    # === CREATE COMBINED FRAMES ===
    combined_frames = []
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame_raw_data_hybird(t, pred, x_offset, height=target_h, width=target_w, annotation_list=annotation_list)

        combined = np.concatenate((middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # === SAVE VIDEO ===
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec='libx264')

def produce_video_raw_data_norm(save_dir, left_video_path, middle_video_path, right_video_path, episode_num, x_offset=30, frame_gap=None):
    # === CONFIGURATION ===
    save_dir = Path(save_dir)
    pred_path = save_dir / "pred.npy"
    conf_path = save_dir / "conf.npy"
    smooth_path = save_dir / "smoothed.npy"
    output_path = save_dir / "combined_video.mp4"
    frame_rate = 30
    
    target_h, target_w = 448, 448  # resolution per panel

    # === LOAD DATA ===
    pred_full = np.load(pred_path)
    pred = pred_full[x_offset:]
    conf = None
    smoothed = None
    if os.path.exists(conf_path):
        conf = np.load(conf_path)[x_offset:]
    if os.path.exists(smooth_path):
        smoothed = np.load(smooth_path)[x_offset:]
    T = len(pred)

    # Load videos
    clip_middle = VideoFileClip(str(middle_video_path))
    
    frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset:]
    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(f"WARNING: Not enough frames in videos. Expected {T}, found {min_frames_num}. Adjusting to available frames.")
        T = min_frames_num
        pred = pred[gap:]
        if conf is not None:
            conf = conf[gap:]
        if smoothed is not None:
            smoothed = smoothed[gap:]
    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    frames_middle = [frames_middle[i] for i in indices]


    # === CREATE COMBINED FRAMES ===
    combined_frames = []
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame_raw_data_norm(t, pred, x_offset, height=target_h, width=target_w, frame_gap=frame_gap, conf=conf, smoothed=smoothed)

        combined = np.concatenate((middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # === SAVE VIDEO ===
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec='libx264')
    
    
def main():
    episode_num = 257
    x_offset = 30
    episode_dir = Path(f"/home/david_chen/rewind_reproduce/outputs/2025-07-16/12-18-19/rewind_reward_fixed_seq_model_frame_gap_multi_stage/pick_up_cube/eval_video/2025.07.16-12.18.26")
    left_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/Qianzhong-Chen/yam_pick_up_cube_sim_rotate_reward_two_stage_0715/videos/chunk-000/left_camera-images-rgb")
    middle_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/Qianzhong-Chen/yam_pick_up_cube_sim_rotate_reward_two_stage_0715/videos/chunk-000/top_camera-images-rgb")
    right_video_dir = Path(f"/home/david_chen/.cache/huggingface/lerobot/Qianzhong-Chen/yam_pick_up_cube_sim_rotate_reward_two_stage_0715/videos/chunk-000/right_camera-images-rgb")
    produce_video(episode_dir, left_video_dir, middle_video_dir, right_video_dir, episode_num, x_offset)

if __name__ == "__main__":
    main()
