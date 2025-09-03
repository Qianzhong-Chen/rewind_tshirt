import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
from functools import lru_cache
from decord import VideoReader, cpu


DATA_PATH = "/nfs_old/david_chen/dataset/tshirt_yam_200_0718/folding_tshirt/episode_z_zFIjhMduwurhFLtp01MJYJv25UttTH9ocB7q9xQic.npy.mp4"

def normalize_sparse(x: float) -> float:
    if 0 <= x < 1:
        return 0.0 + (x - 0) / (1 - 0) * (0.05 - 0.0)
    elif 1 <= x < 2:
        return 0.05 + (x - 1) / (2 - 1) * (0.1 - 0.05)
    elif 2 <= x < 3:
        return 0.1 + (x - 2) / (3 - 2) * (0.3 - 0.1)
    elif 3 <= x < 4:
        return 0.3 + (x - 3) / (4 - 3) * (0.9 - 0.3)
    elif 4 <= x <= 5:
        return 0.9 + (x - 4) / (5 - 4) * (1.0 - 0.9)
    else:
        raise ValueError("x must be in range [0, 5]")
    
def normalize_dense(x: float) -> float:
    if 0 <= x < 1:
        return 0.0 + (x - 0) * (0.08 - 0.0)
    elif 1 <= x < 2:
        return 0.08 + (x - 1) * (0.37 - 0.08)
    elif 2 <= x < 3:
        return 0.37 + (x - 2) * (0.53 - 0.37)
    elif 3 <= x < 4:
        return 0.53 + (x - 3) * (0.67 - 0.53)
    elif 4 <= x <= 5:
        return 0.67 + (x - 4) * (0.72 - 0.67)
    elif 5 <= x <= 6:
        return 0.72 + (x - 5) * (0.81 - 0.72)
    elif 6 <= x <= 7:
        return 0.81 + (x - 6) * (0.9 - 0.81)
    elif 7 <= x <= 8:
        return 0.9 + (x - 7) * (1.0 - 0.9)
    else:
        raise ValueError("x must be in range [0, 8]")



def get_frame_num(path):
    video_path = Path(path) / "top_camera-images-rgb.mp4"
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_joint = Path(path) / "left-joint_pos.npy"
    left_joint_data = np.load(left_joint, allow_pickle=True)
    left_joint_frame, _ = left_joint_data.shape
    left_gripper = Path(path) / "left-gripper_pos.npy"
    left_gripper_data = np.load(left_gripper, allow_pickle=True)
    left_gripper_frame, _ = left_gripper_data.shape
    right_joint = Path(path) / "right-joint_pos.npy"
    right_joint_data = np.load(right_joint, allow_pickle=True)
    right_joint_frame, _ = right_joint_data.shape
    right_gripper = Path(path) / "right-gripper_pos.npy"
    right_gripper_data = np.load(right_gripper, allow_pickle=True)
    right_gripper_frame, _ = right_gripper_data.shape

    frame_num_set = set([total_video_frames, left_joint_frame, left_gripper_frame, right_joint_frame, right_gripper_frame])
    
    if len(frame_num_set) != 1:
        print(f"WARNING: frame number mismatch occures!, Using minimum frames {min(frame_num_set)}")

    return min(frame_num_set)

def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def convert_to_float32(img: np.ndarray) -> np.ndarray:
    """Converts a uint8 image to float32 in [0.0, 1.0] range.

    This is important for restoring the original image scale after transmission.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img

def get_traj_data(path):
    left_joint = Path(path) / "left-joint_pos.npy"
    left_joint_data = np.load(left_joint, allow_pickle=True)
    left_gripper = Path(path) / "left-gripper_pos.npy"
    left_gripper_data = np.load(left_gripper, allow_pickle=True)
    right_joint = Path(path) / "right-joint_pos.npy"
    right_joint_data = np.load(right_joint, allow_pickle=True)
    right_gripper = Path(path) / "right-gripper_pos.npy"
    right_gripper_data = np.load(right_gripper, allow_pickle=True)

    joint_state = np.concatenate((left_joint_data, left_gripper_data[:, 0:1], right_joint_data, right_gripper_data[:, 0:1]), axis=1)
    return joint_state

def get_frame_data(path,
                   traj_joint_data, 
                   idx, 
                   n_obs_steps=6, 
                   frame_gap=15, 
                   max_rewind_steps=4, 
                   camera_names=["top_camera-images-rgb"], 
                   device='cuda:0'):
    
    
    # frames_indices = [0] * (n_obs_steps + 1) # always have the inital frame
    # required_history = n_obs_steps * frame_gap
    # if idx - required_history < 0:
    #     idx = required_history
    # if idx < required_history:
    #     pass
    # else:
    #     frames_indices = [0] + list(reversed([idx - i * frame_gap for i in range(n_obs_steps)]))

    frames_indices = get_frames_indices(idx, n_obs_steps, frame_gap)
    sequence_data = {}

    joint_data = []
    for frame_idx in frames_indices:
        joint_data.append(traj_joint_data[frame_idx, :])
    joint_data = np.array(joint_data)
    joint_padding = np.zeros((max_rewind_steps, joint_data.shape[1]), dtype=np.float32)  # Padding for rewinding
    joint_data = np.concatenate((joint_data, joint_padding), axis=0)  
    sequence_data['state'] = torch.tensor(joint_data, dtype=torch.float32).to(device)
    sequence_data['state'] = sequence_data['state'].unsqueeze(0)

    # process image data
    sequence_data['image_frames'] = {}
    for camera_name in camera_names:
        video_path = Path(path) / f"{camera_name}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max(frames_indices) >= total_video_frames:
            print(f"WARNING: frame index {max(frames_indices)} exceeds total frames {total_video_frames} in {video_path}")
            frames_indices = [total_video_frames - 1] * (n_obs_steps + 1)

        img = []
        for frame_idx in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {video_path}")
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # <-- Fix BGR to RGB
            img.append(frame)

        img = np.array(img)
        img = convert_to_float32(resize_with_pad(img, 224, 224))
        img = np.transpose(img, (0, 3, 1, 2))
        padding_frames = np.zeros((max_rewind_steps, 3, 224, 224), dtype=np.float32)  # Padding frames for rewinding
        img = np.concatenate((img, padding_frames), axis=0)  # Concatenate padding frames at the end
        sequence_data['image_frames'][camera_name] = torch.tensor(img, dtype=torch.float32).to(device)
        sequence_data['image_frames'][camera_name] = sequence_data['image_frames'][camera_name].unsqueeze(0)

    return sequence_data

def get_frames_indices(idx, n_obs_steps, frame_gap):
    """
    Generate frame indices for sequence:
    - Last frame is idx
    - Previous frames spaced roughly by frame_gap
    - Fill with zeros if needed
    """
    # TODO: policy before 0903 need required_history
    required_history = n_obs_steps * frame_gap
    if idx - required_history < 0:
        idx = required_history
    frames = [0] * (n_obs_steps + 1)  # Initialize
    frames[-1] = idx  # last frame

    for i in range(n_obs_steps-1, 0, -1):
        next_frame = frames[i+1] - frame_gap
        frames[i] = max(0, next_frame)

    # Make sure frames are non-decreasing (optional if last 0 should stay)
    for i in range(1, n_obs_steps):
        frames[i] = min(frames[i], frames[i+1])

    return frames

def get_frames_indices_dynamic(idx, n_obs_steps, frame_gap):
    """
    Generate frame indices for a sequence of length n_obs_steps+1 that ends at `idx`.
    - Prefer fixed `frame_gap` when there's enough history.
    - Otherwise, adapt the effective gap by evenly spacing from 0 to `idx`.
    - No zero-padding; indices are clamped to [0, idx] and made non-decreasing.
    """
    idx = int(idx)
    gaps = int(n_obs_steps)  # number of intervals before the last frame

    if gaps <= 0:
        return [idx]

    total_needed = frame_gap * gaps      # framespan needed for fixed gap
    available = idx - 0                  # since we anchor at 0 here

    if available >= total_needed:
        # Use fixed frame_gap
        frames = [idx - frame_gap * (gaps - k) for k in range(gaps)] + [idx]
    else:
        # Not enough history: evenly space from 0..idx (adaptive gap)
        frames = [round(available * k / gaps) for k in range(gaps)] + [idx]

    # Clamp and enforce non-decreasing
    frames = [max(0, min(idx, f)) for f in frames]
    for i in range(1, len(frames)):
        if frames[i] < frames[i - 1]:
            frames[i] = frames[i - 1]
    return frames

@lru_cache(maxsize=64)
def _get_vr(video_path: str):
    # CPU decode; switch to decord.gpu(0) if you want GPU decoding later
    return VideoReader(video_path, ctx=cpu(0))

def get_frame_data_fast(path,
                   traj_joint_data, 
                   idx, 
                   n_obs_steps=6, 
                   frame_gap=15, 
                   max_rewind_steps=4, 
                   camera_names=("top_camera-images-rgb",), 
                   device='cuda:0'):
    """
    Exact same output as your original:
    {
        'state': FloatTensor [1, n_obs_steps+1+max_rewind_steps, joint_dim],
        'image_frames': { cam: FloatTensor [1, n_obs_steps+1+max_rewind_steps, 3, 224, 224], ... }
    }
    Assumes your existing `get_frames_indices`, `resize_with_pad`, `convert_to_float32`.
    """
    # --- frame indices (unchanged) ---
    frames_indices = get_frames_indices(idx, n_obs_steps, frame_gap)
    sequence_data = {}

    # --- joints (unchanged logic) ---
    joint_data = np.array([traj_joint_data[i, :] for i in frames_indices], dtype=np.float32)
    if max_rewind_steps > 0:
        joint_padding = np.zeros((max_rewind_steps, joint_data.shape[1]), dtype=np.float32)
        joint_data = np.concatenate((joint_data, joint_padding), axis=0)
    sequence_data['state'] = torch.tensor(joint_data, dtype=torch.float32, device=device).unsqueeze(0)

    # --- images via decord, but keep exact post-processing (resize_with_pad -> convert_to_float32 -> NCHW -> pad -> unsqueeze) ---
    sequence_data['image_frames'] = {}
    for camera_name in camera_names:
        video_path = str(Path(path) / f"{camera_name}.mp4")
        vr = _get_vr(video_path)
        total_video_frames = len(vr)

        # match original edge handling: if any index too large, repeat last frame for all
        if max(frames_indices) >= total_video_frames:
            print(f"WARNING: frame index {max(frames_indices)} exceeds total frames {total_video_frames} in {video_path}")
            safe_indices = [total_video_frames - 1] * (n_obs_steps + 1)
        else:
            safe_indices = frames_indices

        # batched random access, RGB, (T, H, W, 3), uint8
        batch = vr.get_batch(safe_indices)
        img = batch.asnumpy()  # (T,H,W,3), uint8, RGB

        # keep your exact preprocessing calls
        img = convert_to_float32(resize_with_pad(img, 224, 224))  # expects (T,H,W,3); returns float32 in [0,1]

        # NCHW
        img = np.transpose(img, (0, 3, 1, 2))  # (T,3,224,224)

        # pad rewind frames (zeros), same as before
        if max_rewind_steps > 0:
            padding_frames = np.zeros((max_rewind_steps, 3, 224, 224), dtype=np.float32)
            img = np.concatenate((img, padding_frames), axis=0)

        tensor = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T+rewind,3,224,224)
        sequence_data['image_frames'][camera_name] = tensor

    return sequence_data

if __name__ == "__main__":
    # get_frame_num(DATA_PATH)
    # traj_joint_data = get_traj_data(DATA_PATH)
    # sequence_data = get_frame_data(DATA_PATH, traj_joint_data, idx=200)
    # import pdb; pdb.set_trace()
    # print(sequence_data)
    
    print(get_frames_indices(13,3,12))