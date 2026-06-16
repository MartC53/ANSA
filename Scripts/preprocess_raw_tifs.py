# %%
import os
import numpy as np
import cv2
from skimage import io


# %%
def background_sub(data: np.ndarray, n_bg: int = 20) -> np.ndarray:
    """
    Subtract a background from each frame of the stack, using the average
    of the first 'n_bg' frames as the background reference.

    Parameters
    ----------
    data : np.ndarray
        3D image stack with shape (num_frames, height, width).
    n_bg : int, optional
        Number of frames to average for background calculation, by default 20.

    Returns
    -------
    np.ndarray
        Background-subtracted stack (type uint16, nonnegative).
    """
    num_frames, height, width = data.shape
    background = np.zeros((height, width), dtype=np.float64)

    # Average the first n_bg frames
    for i in range(min(n_bg, num_frames)):
        background += data[i]
    background /= min(n_bg, num_frames)

    # Subtract background from all frames
    data_adj = data.astype(np.float64) - background
    data_adj[data_adj < 0] = 0  # Clip negative values to zero
    return data_adj.astype(np.uint16)


def gaussian_blur(data: np.ndarray, ksize: int = 7) -> np.ndarray:
    """
    Apply a 2D Gaussian blur to each frame in the stack.

    Parameters
    ----------
    data : np.ndarray
        3D image stack with shape (num_frames, height, width).
    ksize : int, optional
        Kernel size for the Gaussian blur, by default 7.

    Returns
    -------
    np.ndarray
        The blurred image stack.
    """
    blurred_stack = []
    for i in range(data.shape[0]):
        blurred_frame = cv2.GaussianBlur(data[i], (ksize, ksize), cv2.BORDER_DEFAULT)
        blurred_stack.append(blurred_frame)

    return np.array(blurred_stack, dtype=data.dtype)


def rolling_ball_filter(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply a rolling-ball filter in the time dimension by averaging 
    neighboring frames and subtracting from the current frame.

    Parameters
    ----------
    data : np.ndarray
        3D stack with shape (num_frames, height, width).
    window_size : int, optional
        Number of frames on either side of the current frame to average,
        by default 3.

    Returns
    -------
    np.ndarray
        The processed stack, same shape as input.
    """
    num_frames, height, width = data.shape
    processed = np.zeros_like(data, dtype=np.float32)

    for i in range(num_frames):
        start_index = max(0, i - window_size)
        end_index = min(num_frames, i + window_size + 1)
        avg = np.mean(data[start_index:end_index], axis=0)
        processed[i] = data[i].astype(np.float32) - avg

    processed[processed < 0] = 0
    return processed.astype(data.dtype)


def circle_mask(image: np.ndarray, shrink_factor: float = 0.90) -> np.ndarray:
    """
    Apply a circular mask to a single 2D image, zeroing out pixels 
    outside the circle.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (height x width).
    shrink_factor : float, optional
        Fraction of the largest possible radius to use (default is 0.90).

    Returns
    -------
    np.ndarray
        The masked image, same shape as input.
    """
    height, width = image.shape
    center = (width // 2, height // 2)
    max_radius = min(center)
    radius = int(max_radius * shrink_factor)

    y, x = np.ogrid[:height, :width]
    r2 = (x - center[0])**2 + (y - center[1])**2
    mask = r2 <= radius**2

    return np.where(mask, image, 0)


def process_directory(
    input_dir: str, 
    output_dir: str, 
    n_bg: int = 25, 
    window_size: int = 3
) -> None:
    """
    Process all .tif stacks in input_dir with the following pipeline:
    1) Background subtraction
    2) Gaussian blur
    3) Rolling-ball filter
    4) Circle mask
    Then save them to output_dir.

    Parameters
    ----------
    input_dir : str
        Path to the directory with input .tif files (each assumed to be [frames, height, width]).
    output_dir : str
        Path to the directory for saving the processed files.
    n_bg : int, optional
        Number of frames to average for background subtraction, default 25.
    window_size : int, optional
        Half-width for rolling-ball filter averaging, default 3.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith('.tif'):
            continue
        file_path = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        # Load
        data_stack = io.imread(file_path)  # shape: [frames, H, W]

        # 1) Background subtraction
        data_stack = background_sub(data_stack, n_bg=n_bg)

        # 2) Gaussian blur
        data_stack = gaussian_blur(data_stack)

        # 3) Rolling-ball filter (time domain)
        data_stack = rolling_ball_filter(data_stack, window_size=window_size)

        # 4) Apply circle mask on each frame
        for i in range(data_stack.shape[0]):
            data_stack[i] = circle_mask(data_stack[i])

        # Save result
        save_path = os.path.join(output_dir, filename)
        io.imsave(save_path, data_stack, check_contrast=False)
        print(f"Saved processed stack to: {save_path}")


if __name__ == "__main__":
    # Example usage:
    input_folder = r"M:\data"       # Replace with your own path
    output_folder = r"M:\data"      # Replace with your own path

    process_directory(input_folder, output_folder, n_bg=25, window_size=3)



