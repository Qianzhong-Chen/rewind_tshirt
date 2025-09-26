import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from moviepy.editor import VideoFileClip, ImageSequenceClip
from pathlib import Path
from raw_data_utils import resize_with_pad
import os
from matplotlib.patches import ConnectionPatch


def center_crop_to_848x480(img: np.ndarray, tw: int = 848, th: int = 480) -> np.ndarray:
    """Center-crop to ~16:9 (848x480) then resize to exactly (848,480)."""
    h, w = img.shape[:2]
    target_ar = tw / th
    ar = w / h
    if ar > target_ar:  # too wide -> crop width
        new_w = int(h * target_ar)
        x0 = (w - new_w) // 2
        crop = img[:, x0:x0 + new_w]
    else:               # too tall -> crop height
        new_h = int(w / target_ar)
        y0 = (h - new_h) // 2
        crop = img[y0:y0 + new_h, :]
    return cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

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

# def draw_plot_frame_raw_data(step: int, pred, x_offset, width=448, height=448):
#     fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448

#     timesteps = np.arange(len(pred)) + x_offset
#     ax.plot(timesteps, pred, label='Predicted', linewidth=2)
#     ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
#     ax.set_title("Reward Model Prediction")
#     ax.set_xlabel("Time Step")
#     ax.set_ylabel("Reward")
#     ax.legend()
#     ax.grid(True)
#     fig.tight_layout()

#      # === Add Milestone Text ===
#     current_pred = pred[step]
#     text_y = ax.get_ylim()[1] * 0.9  # position near top

#     # Sparse anno
#     # if current_pred <= 1.0:
#     #     ax.text(step + x_offset, text_y, "Grab the tshirt from the pile", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     # elif 1.0 <= current_pred < 2.0:
#     #     ax.text(step + x_offset, text_y, "Move the tshirt to the center of the board", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     # elif 2.0 <= current_pred < 3.0:
#     #     ax.text(step + x_offset, text_y, "Flatten the tshirt out", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     # elif 3.0 <= current_pred < 4.0:
#     #     ax.text(step + x_offset, text_y, "Fold the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     # elif 4.0 <= current_pred < 5.0:
#     #     ax.text(step + x_offset, text_y, "Neatly place the folded tshirt to the corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     # else:
#     #     ax.text(step + x_offset, text_y, "Task Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
    
#     # Dense anno
#     annotation_list= [
#             "grab crumpled tshirt and move to center",
#             "flatten out the tshirt",
#             "grab near side and fold one-third",
#             "grab far side and fold into rectangle",
#             "rotate the tshirt 90 degrees",
#             "grab bottom and fold one-third",
#             "grab two-third side and fold into square",
#             "put folded tshirt into corner"
#         ]
#     if current_pred <= 1.0:
#         ax.text(step + x_offset, text_y, "grab crumpled tshirt and move to center", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 1.0 <= current_pred < 2.0:
#         ax.text(step + x_offset, text_y, "flatten out the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 2.0 <= current_pred < 3.0:
#         ax.text(step + x_offset, text_y, "grab near side and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 3.0 <= current_pred < 4.0:
#         ax.text(step + x_offset, text_y, "grab far side and fold into rectangle", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 4.0 <= current_pred < 5.0:
#         ax.text(step + x_offset, text_y, "rotate the tshirt 90 degrees", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 5.0 <= current_pred < 6.0:
#         ax.text(step + x_offset, text_y, "grab bottom and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 6.0 <= current_pred < 7.0:
#         ax.text(step + x_offset, text_y, "grab two-third side and fold into square", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 7.0 <= current_pred < 8.0:
#         ax.text(step + x_offset, text_y, "put folded tshirt into corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     else:
#         ax.text(step + x_offset, text_y, "Task Finished", color='green', fontsize=12, fontweight='bold', ha='center', va='top')


#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
#     img = img.reshape(canvas.get_width_height()[::-1] + (4,))
#     img = img[:, :, :3]  # get RGB
#     plt.close(fig)

#     return img

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
        timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset * frame_gap

    # === Plot raw prediction ===
    pred = smoothed if smoothed is not None else pred
    line_pred, = ax.plot(timesteps, pred, label='Predicted', linewidth=2)
    handles, labels = [line_pred], ["Predicted"]

    # # === Plot smoothed prediction ===
    # if smoothed is not None:
    #     line_smooth, = ax.plot(timesteps, smoothed, label="Smoothed", linewidth=2, color="orange")
    #     handles.append(line_smooth)
    #     labels.append("Smoothed")

    # === Vertical line at current step ===
    if frame_gap is None:
        ax.axvline(x=step + x_offset, color='r', linestyle='--', linewidth=2)
    else:
        ax.axvline(x=step * frame_gap + x_offset * frame_gap, color='r', linestyle='--', linewidth=2)

    # === Labels and style ===
    ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Reward")
    ax.grid(True)

    # # === Confidence curve on twin axis ===
    # if conf is not None:
    #     ax2 = ax.twinx()
    #     line_conf, = ax2.plot(timesteps, conf, linestyle=':', color='green', label="Confidence")
    #     ax2.set_ylabel("Confidence")
    #     handles.append(line_conf)
    #     labels.append("Confidence")

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


def piecewise_transform(arr: np.ndarray) -> np.ndarray:
    """
    Apply piecewise transformation to a 1D numpy array.
    
    Rules:
      - If 0 <= x < 0.6: f(x) = x / 0.6
      - If 0.6 <= x < 0.7: f(x) = (x - 0.6) * 2.5 + 0.5
      - If 0.7 <= x <= 1: f(x) = (x - 0.7) + 0.75
    """
    result = np.zeros_like(arr, dtype=float)
    
    mask1 = (arr >= 0) & (arr < 0.6)
    mask2 = (arr >= 0.6) & (arr < 0.7)
    mask3 = (arr >= 0.7) & (arr <= 1.0)
    
    result[mask1] = arr[mask1] * 0.833
    result[mask2] = (arr[mask2] - 0.6) * 4.0 + 0.5
    result[mask3] = np.minimum((arr[mask3] - 0.7) + 0.90, 1.0)
    
    return result

# # new transform for plot with gt, ep_21 special
# def piecewise_transform(arr: np.ndarray) -> np.ndarray:
#     """
#     Apply piecewise transformation to a 1D numpy array.
    
#     Rules:
#       - If 0 <= x < 0.6: f(x) = x / 0.6
#       - If 0.6 <= x < 0.7: f(x) = (x - 0.6) * 2.5 + 0.5
#       - If 0.7 <= x <= 1: f(x) = (x - 0.7) + 0.75
#     """
#     result = arr.copy()
#     result[38:49] = np.minimum((arr[38:49] - 0.2937)*0.75 +0.2937, 1.0)
#     result[49:60] = np.minimum((arr[49:60]-0.05), 1.0)
#     result[60:100] = np.minimum((arr[60:100] - 0.57)*2.0 +0.57, 1.0)
#     # mask1 = (arr >= 0.57) & (arr <= 0.71)
#     mask2 = (arr > 0.71) & (arr <= 1.0)
#     # result[mask1] = np.minimum((arr[mask1] - 0.57)*2.0 +0.57, 1.0)
#     result[mask2] = np.minimum((arr[mask2] - 0.71)*0.85 + 0.85, 1.0)

#     return result


# success rollout
def piecewise_transform_raw(arr: np.ndarray) -> np.ndarray:
    result = arr.copy()
    mask1 = (arr >= 0.48) & (arr <= 1.0)
    result[mask1] = np.minimum((arr[mask1] - 0.48)*2.5 + 0.48, 1.0)
    
    return result

# # fake finish
# def piecewise_transform_raw(arr: np.ndarray) -> np.ndarray:
#     result = arr.copy()
#     mask1 = (arr >= 0.7) & (arr <= 1.0)
#     result[mask1] = np.minimum((arr[mask1] - 0.7)* 3.5 + 0.7, 1.0)
    
#     return result


def draw_plot_frame_raw_data_hybird(step: int, pred, x_offset, width=448, height=448, frame_gap=None, conf=None, smoothed=None):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448
    pred = smoothed if smoothed is not None else pred
    if frame_gap is None:
        timesteps = np.arange(len(pred)) + x_offset
    else:
        timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset * frame_gap
        
    # === Plot raw prediction ===
    real_time  = timesteps * 0.033  # assuming 30 FPS, convert to seconds
    line_pred, = ax.plot(real_time, pred, label='Predicted', linewidth=2)
    handles, labels = [line_pred], ["Predicted"]

    # === Vertical line at current step ===
    if frame_gap is None:
        ax.axvline(x=(step + x_offset)*0.033, color='r', linestyle='--', linewidth=2)
    else:
        ax.axvline(x=(step * frame_gap + x_offset * frame_gap)*0.033, color='r', linestyle='--', linewidth=2)

    # === Labels and style ===
    # ax.set_title("Reward Model Prediction")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Progress", fontsize=18)
    ax.grid(True)

    # === Legend ===
    # ax.legend(handles, labels, loc="best")

    fig.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # get RGB
    plt.close(fig)

    return img

# # Ep 21
# def draw_plot_frame_raw_data_hybird(step: int, pred, x_offset, width=448, height=448, frame_gap=None, conf=None, smoothed=None):
#     fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448
#     pred = smoothed if smoothed is not None else pred
#     if frame_gap is None:
#         timesteps = np.arange(len(pred)) + x_offset
#     else:
#         timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset * frame_gap
        
#     # === Plot raw prediction ===
#     real_time  = timesteps * 0.033  # assuming 30 FPS, convert to seconds
#     line_pred, = ax.plot(real_time, pred, label='Predicted', linewidth=2)
#     handles, labels = [line_pred], ["progress"]

#     # === Vertical line at current step ===
#     if frame_gap is None:
#         ax.axvline(x=(step + x_offset)*0.033, color='r', linestyle='--', linewidth=2)
#     else:
#         ax.axvline(x=(step * frame_gap + x_offset * frame_gap)*0.033, color='r', linestyle='--', linewidth=2)

#     # === Labels and style ===
#     # ax.set_title("Reward Model Prediction")
#     ax.set_xlabel("Time (s)", fontsize=18)
#     ax.set_ylabel("Progress", fontsize=18)
#     ax.grid(True)

#     # === Legend ===
#     # ax.legend(handles, labels, loc="best", fontsize=18)

#     fig.tight_layout()

#     # === Add Milestone Text ===
#     text_y = ax.get_ylim()[1] * 0.9  # position near top

#     # Sparse anno
#     text_x_pos = 20
#     annotation_list= ["Grab the tshirt from the pile", 
#                       "Move the tshirt to the center of the board",
#                       "Flatten the tshirt out",
#                       "Fold the tshirt",
#                       "Neatly place the folded tshirt to the corner",
#                       "task finished"]
#     if step <= 30:
#         ax.text(text_x_pos, text_y, "Grab the tshirt from the pile", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     # elif 18 <= step < 30:
#     #     ax.text(text_x_pos, text_y, "Move the tshirt to the center of the board", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 30 <= step < 78:
#         ax.text(text_x_pos, text_y, "Flatten the tshirt out", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 78 <= step < 198:
#         ax.text(text_x_pos, text_y, "Fold the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     else:
#         ax.text(text_x_pos, text_y, "Put folded tshirt into corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')

    
#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
#     img = img.reshape(canvas.get_width_height()[::-1] + (4,))
#     img = img[:, :, :3]  # get RGB
#     plt.close(fig)

#     return img

# # Ep 320
# def draw_plot_frame_raw_data_hybird(step: int, pred, x_offset, width=448, height=448, frame_gap=None, conf=None, smoothed=None):
#     fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # ensures final image is 448x448
#     pred = smoothed if smoothed is not None else pred
#     if frame_gap is None:
#         timesteps = np.arange(len(pred)) + x_offset
#     else:
#         timesteps = np.arange(0, len(pred) * frame_gap, frame_gap) + x_offset * frame_gap
        
#     # === Plot raw prediction ===
#     real_time  = timesteps * 0.033  # assuming 30 FPS, convert to seconds
#     line_pred, = ax.plot(real_time, pred, label='Predicted', linewidth=2)
#     handles, labels = [line_pred], ["progress"]

#     # === Vertical line at current step ===
#     if frame_gap is None:
#         ax.axvline(x=(step + x_offset)*0.033, color='r', linestyle='--', linewidth=2)
#     else:
#         ax.axvline(x=(step * frame_gap + x_offset * frame_gap)*0.033, color='r', linestyle='--', linewidth=2)

#     # === Labels and style ===
#     # ax.set_title("Reward Model Prediction")
#     ax.set_xlabel("Time (s)", fontsize=18)
#     ax.set_ylabel("Progress", fontsize=18)
#     ax.grid(True)

#     # === Legend ===
#     # ax.legend(handles, labels, loc="best", fontsize=18)

#     fig.tight_layout()

#     # === Add Milestone Text ===
#     text_y = ax.get_ylim()[1] * 0.9  # position near top

#     # Dense anno
#     text_x_pos = 21
#     annotation_list= [
#             "grab crumpled tshirt and move to center",
#             "flatten out the tshirt",
#             "grab near side and fold one-third",
#             "grab far side and fold into rectangle",
#             "rotate the tshirt 90 degrees",
#             "grab bottom and fold one-third",
#             "grab two-third side and fold into square",
#             "put folded tshirt into corner"
#     ]
    
#     if step <= 24:
#         ax.text(text_x_pos, text_y, "grab crumpled tshirt and move to center", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 24 <= step < 66:
#         ax.text(text_x_pos, text_y, "flatten out the tshirt", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 66 <= step < 120:
#         ax.text(text_x_pos, text_y, "grab near side and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 120 <= step < 150:
#         ax.text(text_x_pos, text_y, "grab far side and fold into rectangle", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 150 <= step < 174:
#         ax.text(text_x_pos, text_y, "rotate the tshirt 90 degrees", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 174 <= step < 192:
#         ax.text(text_x_pos, text_y, "grab bottom and fold one-third", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     elif 192 <= step < 228:
#         ax.text(text_x_pos, text_y, "grab two-third side and fold into square", color='green', fontsize=12, fontweight='bold', ha='center', va='top')
#     else:
#         ax.text(text_x_pos, text_y, "put folded tshirt into corner", color='green', fontsize=12, fontweight='bold', ha='center', va='top')

#     canvas = FigureCanvas(fig)
#     canvas.draw()
#     img = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').copy()
#     img = img.reshape(canvas.get_width_height()[::-1] + (4,))
#     img = img[:, :, :3]  # get RGB
#     plt.close(fig)

#     return img

def draw_overview_panel(frames: list[np.ndarray],
                        reward: np.ndarray,
                        frame_rate: float,
                        save_path: str = "overview_panel.pdf",
                        out_w_each: int = 848,
                        out_h_each: int = 480) -> np.ndarray:
    """
    Top: 4 cropped images side-by-side (5th, 1/4, ~0.7T, end-3).
    Bottom: reward vs time (seconds), 1.5x taller than image row.
    Red connectors from each image to its reward location.
    Saves figure as a PDF with Times New Roman text.
    Also returns the rendered RGB array.
    """
    import matplotlib as mpl

    
    T = len(frames)
    if T < 10 or len(reward) < 2:
        raise ValueError("Not enough frames/reward to build overview panel.")

    # --- SET FONT TO TIMES NEW ROMAN ---
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 42
    mpl.rcParams['pdf.fonttype'] = 42  # embed fonts in PDF
    
    # demo
    idxs = [min(4, T-1), max(0, min(T//4, T-1)),
            max(0, min(int(0.7*T), T-1)), max(0, T-3)]
    
    # # rollout
    # idxs = [max(0, min(T//4, T-1)), max(0, min(int(0.61*T), T-1)),
    #         max(0, min(int(0.70*T), T-1)), max(0, T-20)]

    # crop thumbnails to 848x480
    thumbs = [center_crop_to_848x480(frames[i]) for i in idxs]

    # time axis (sec)
    t = np.arange(len(reward), dtype=float) / float(frame_rate) * 10

    # --- Figure/layout ---
    total_h_units = 1.0 + 1.5
    fig_w = (out_w_each * 4) / 100.0
    fig_h = (out_h_each * total_h_units) / 100.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=4,
                          height_ratios=[1.0, 1.5],
                          hspace=0.02, wspace=0.01)

    top_axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ax = fig.add_subplot(gs[1, :])

    # font sizes
    label_fs, tick_fs = 50, 46

    # --- show thumbnails ---
    for ax_img, im in zip(top_axes, thumbs):
        ax_img.imshow(im)
        ax_img.set_xticks([]); ax_img.set_yticks([])

    # --- reward plot ---
    ax.plot(t, reward, linewidth=6)
    ylim = ax.get_ylim()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=label_fs)
    ax.set_ylabel("Predicted Progress", fontsize=label_fs)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fs)

    # --- connectors & markers ---
    for i, idx in enumerate(idxs):
        xi = t[min(idx, len(t)-1)]
        yi = float(reward[min(idx, len(reward)-1)])

        ax.plot([xi], [yi], 'o', color='r', markersize=32)

        con = ConnectionPatch(
            xyA=(0.5, 0.0), coordsA=top_axes[i].transAxes,
            xyB=(xi, yi),   coordsB=ax.transData,
            arrowstyle='-',
            linestyle="--",
            color='r', linewidth=3.0, alpha=0.9
        )
        fig.add_artist(con)

    # --- save to PDF ---
    fig.savefig(save_path, format="pdf", bbox_inches="tight")

    # --- also return as array ---
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img

def draw_overview_panel_dual(
    frames: list[np.ndarray],
    reward_main: np.ndarray,
    frame_rate: float,
    alt_curve_npy: str | None = None,
    reward_alt: np.ndarray | None = None,
    save_path: str = "overview_panel.pdf",
    out_w_each: int = 848,
    out_h_each: int = 480,
    dash_pad: float = 0.05,          # extra margin added above/below the two-curve span
    dash_min_height: float = 0.12,   # if curves are too close, enforce at least this height
    dash_ls: str = "--",
    dash_lw: float = 6.0,
    conn_lw: float = 4.0,
    guide_color: str = "tab:red", # vertical dashed segment color
    conn_color: str = "tab:red",  # connector line color
):
    """
    Top: 4 thumbnails; Bottom: reward_main (+ optional reward_alt) vs time.
    For each chosen timestamp:
      - draw an adaptive-height vertical dashed line at x = timestamp that spans both curves (+ padding)
      - draw a connector from the bottom-center of the thumbnail to the dashed segment midpoint.
    Saves a PDF and returns the rendered RGB array.

    If `alt_curve_npy` is provided, it is loaded and overrides `reward_alt`.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.patches import ConnectionPatch

    # ---- validate & load second curve if needed ----
    T = len(frames)
    if T < 10 or len(reward_main) < 2:
        raise ValueError("Not enough frames/reward to build overview panel.")

    if alt_curve_npy is not None:
        reward_alt = np.load(alt_curve_npy)
        
    gt_alt = np.load("/nfs_us/david_chen/reward_model_ckpt/tshirt_rollout/2025-09-21/22-03-33/fold_tshirt_hybird/eval_video/episode_21/gt.npy")

    # ---- fonts (Times New Roman) ----
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.size'] = 42
    mpl.rcParams['pdf.fonttype'] = 42

    # ---- pick 4 indices (you can swap to your rollout set if you prefer) ----
    idxs = [
        min(4, T - 1),
        max(0, min(T // 4, T - 1)),
        max(0, min(int(0.7 * T), T - 1)),
        max(0, T - 3),
    ]

    # center-crop helper must exist in your codebase
    thumbs = [center_crop_to_848x480(frames[i]) for i in idxs]

    # ---- unify lengths for overlay ----
    if reward_alt is not None:
        L = min(len(reward_main), len(reward_alt))
        r1 = reward_main[:L]
        r2 = reward_alt[:L]
    else:
        L = len(reward_main)
        r1 = reward_main
        r2 = None

    # time in seconds
    t = np.arange(L, dtype=float) / float(frame_rate) * 10

    # map frame indices -> reward indices (proportional)
    ridxs = [int(round(i / max(T - 1, 1) * (L - 1))) for i in idxs]

    # ---- figure & layout ----
    total_h_units = 1.0 + 1.5
    fig_w = (out_w_each * 4) / 100.0
    fig_h = (out_h_each * total_h_units) / 100.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=4, height_ratios=[1.0, 1.5], hspace=0.02, wspace=0.01)
    top_axes = [fig.add_subplot(gs[0, c]) for c in range(4)]
    ax = fig.add_subplot(gs[1, :])

    # ---- top thumbnails ----
    for ax_img, im in zip(top_axes, thumbs):
        ax_img.imshow(im)
        ax_img.set_xticks([]); ax_img.set_yticks([])

    # ---- bottom curves ----
    label_fs, tick_fs = 50, 50
    lw = 6

    r0 = gt_alt[:L]
    ax.plot(t, r0, linewidth=lw, label="Ground Truth", linestyle=":", color="gray")
    ax.plot(t, r1, linewidth=lw, label="Proposed Model")
    
    if r2 is not None:
        ax.plot(t, r2, linewidth=lw, label="ReWiND Model")
        ax.legend(fontsize=38, frameon=False, loc="lower right")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (s)", fontsize=label_fs)
    ax.set_ylabel("Predicted Progress", fontsize=label_fs)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fs)

    y_min_ax, y_max_ax = ax.get_ylim()

    # ---- vertical dashed segments + connectors ----
    for i, ridx in enumerate(ridxs):
        xi = float(t[ridx])
        vals = [float(r1[ridx])]
        if r2 is not None:
            vals.append(float(r2[ridx]))

        vmin = min(vals)
        vmax = max(vals)

        # enforce a minimum height with symmetric padding
        span = max(vmax - vmin + 2 * dash_pad, dash_min_height)
        y0 = (vmin + vmax) / 2.0 - span / 2.0
        y1 = (vmin + vmax) / 2.0 + span / 2.0

        # clamp to axis limits
        y0 = max(y0, y_min_ax)
        y1 = min(y1, y_max_ax)

        # vertical dashed line at x = xi
        ax.vlines(xi, y0, y1, linestyles=dash_ls, linewidth=dash_lw, color=guide_color, alpha=0.9)

        # connector from thumbnail bottom-center to the midpoint of the dashed segment
        y_mid = (y0 + y1) / 2.0
        con = ConnectionPatch(
            xyA=(0.5, 0.0), coordsA=top_axes[i].transAxes,   # bottom center of image axes
            xyB=(xi, y_mid),   coordsB=ax.transData,         # midpoint of dashed segment
            arrowstyle='-', linestyle='-', linewidth=conn_lw, color=conn_color, alpha=0.95
        )
        fig.add_artist(con)

    # ---- save & return RGB array ----
    fig.savefig(save_path, format="pdf", bbox_inches="tight")

    canvas = FigureCanvas(fig)
    canvas.draw()
    import numpy as np
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).copy()
    img = img.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return img


def produce_video(save_dir, left_video_dir, middle_video_dir, right_video_dir, episode_num, x_offset=30, frame_gap=None):
    frame_gap = 5
    # === CONFIGURATION ===
    episode_dir = save_dir / f"episode_{episode_num}"
    left_video_path = left_video_dir / f"episode_{episode_num:06d}.mp4"
    middle_video_path = middle_video_dir / f"episode_{episode_num:06d}.mp4"
    right_video_path = right_video_dir / f"episode_{episode_num:06d}.mp4"
    pred_path = episode_dir / "pred.npy"
    conf_path = episode_dir / "conf.npy"
    smooth_path = episode_dir / "smoothed.npy"
    gt_path = episode_dir / "gt.npy"
    output_path = episode_dir / "combined_video.mp4"
    frame_rate = 30

    target_h, target_w = 448, 448  # resolution per panel

    # === LOAD DATA ===
    pred_full = np.load(pred_path)
    gt_full = np.load(gt_path)
    pred = pred_full[x_offset:]
    gt = gt_full[x_offset:]
    conf = None
    smoothed = pred
    if os.path.exists(conf_path):
        conf = np.load(conf_path)[x_offset:]
    if os.path.exists(smooth_path):
        smoothed = np.load(smooth_path)[x_offset:]
    T = len(pred)*4

    # Load videos
    clip_middle = VideoFileClip(str(middle_video_path))
    if frame_gap is not None:
        frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset*frame_gap:]
    else:
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
    smoothed = piecewise_transform(smoothed)
    orig_idx = np.arange(len(smoothed))
    # Target indices (double size)
    new_idx = np.linspace(0, len(smoothed) - 1, 4 * len(smoothed) - 1)
    # Interpolated result
    smoothed = np.interp(new_idx, orig_idx, smoothed)
    
    
    # overview = draw_overview_panel(frames_middle, smoothed, frame_rate, save_path=str(episode_dir / "overview_panel.pdf"))
    # overview = draw_overview_panel_dual(frames_middle, smoothed, frame_rate, reward_alt=gt, save_path=str(episode_dir / "overview_panel_dual.pdf"), 
    #                                     alt_curve_npy="/nfs_us/david_chen/reward_model_ckpt/tshirt_rollout/2025-09-18/11-52-04/fold_tshirt_regression_sparse/eval_video/2025.09.18-11.52.18/episode_21/smoothed.npy")
    # # save alongside the episode video
    # summary_path = episode_dir / "overview_panel.png"
    # # cv2 expects BGR
    # cv2.imwrite(str(summary_path), overview[:, :, ::-1])

    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame_raw_data_hybird(t, pred, x_offset, height=target_h, width=target_w, frame_gap=frame_gap, conf=conf, smoothed=smoothed)

        combined = np.concatenate((middle_resized, plot_img), axis=1)
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
    overview = draw_overview_panel(frames_middle, pred, frame_rate, save_path=str(output_path / "overview_panel.pdf"))
    # save alongside the episode video
    summary_path = output_path / "overview_panel.png"
    # cv2 expects BGR
    cv2.imwrite(str(summary_path), overview[:, :, ::-1])
    
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame_raw_data(t, pred, x_offset, height=target_h, width=target_w)

        combined = np.concatenate((middle_resized, plot_img), axis=1)
        combined_frames.append(combined)

    # === SAVE VIDEO ===
    output_clip = ImageSequenceClip(combined_frames, fps=frame_rate)
    output_clip.write_videofile(str(output_path), codec='libx264')
    
def produce_video_raw_data_hybird(save_dir, left_video_path, middle_video_path, right_video_path, episode_num, annotation_list=None, x_offset=30, frame_gap=None):
    frame_gap = 5
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
    else:
        smoothed = pred
    T = len(pred) * 4

    # Load videos
    clip_middle = VideoFileClip(str(middle_video_path))
    if frame_gap is not None:
        frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset*frame_gap:]
    else:
        frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset:]
    min_frames_num = len(frames_middle)
    if min_frames_num < T:
        gap = T - min_frames_num
        print(f"WARNING: Not enough frames in videos. Expected {T}, found {min_frames_num}. Adjusting to available frames.")
        T = min_frames_num
        pred = pred[gap:]
    total_len = len(frames_middle)
    indices = np.linspace(0, total_len - 1, T, dtype=int)
    # frames_middle = [frames_middle[i] for i in indices]
    frames_middle = np.asarray(frames_middle)        
    frames_middle = resize_with_pad(frames_middle, 224, 224)
    frames_middle = [frames_middle[i] for i in indices]


    # === CREATE COMBINED FRAMES ===
    combined_frames = []
    orig_idx = np.arange(len(smoothed))
    new_idx = np.linspace(0, len(smoothed) - 1, 4 * len(smoothed) - 1)
    smoothed = np.interp(new_idx, orig_idx, smoothed)
    
    smoothed = piecewise_transform_raw(smoothed)
   
    # overview = draw_overview_panel(frames_middle, smoothed, frame_rate, save_path=str(save_dir / "overview_panel.pdf"))
    # summary_path = save_dir / "overview_panel.png"
    # cv2.imwrite(str(summary_path), overview[:, :, ::-1])
    
    for t in range(T):
        middle_resized = cv2.resize(frames_middle[t], (target_w, target_h))
        plot_img = draw_plot_frame_raw_data_hybird(t, pred, x_offset, height=target_h, width=target_w, frame_gap=frame_gap, conf=conf, smoothed=smoothed)

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
    
    if frame_gap is not None:
        frames_middle = [f for f in clip_middle.iter_frames(fps=frame_rate)][x_offset*frame_gap:]
    else:
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
