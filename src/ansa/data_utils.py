"""Data utility functions for ANSA model preprocessing and dataset inspection.

This module contains reusable, path-configurable helpers extracted from the
original `Accessory_functions.py`. It intentionally does not execute any local
file-system operations at import time.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "0.undetectable",
    "1.low",
    "2.medium",
    "3.high",
    "4.very high",
)

DEFAULT_SELECTED_FRAME_INDICES: tuple[int, ...] = (69, 89, 109, 129, 149, 179)


def estimate_dataset_size_gb(dataset) -> float:
    """Estimate the in-memory size of a PyTorch-style dataset in GB.

    The dataset is expected to return `(sample, label)` pairs where `sample` is a
    tensor. The estimate assumes every sample has the same shape and dtype as the
    first sample.
    """
    if len(dataset) == 0:
        return 0.0

    sample, _ = dataset[0]
    if not torch.is_tensor(sample):
        raise TypeError("Expected dataset[0][0] to be a torch.Tensor.")

    sample_size_bytes = sample.element_size() * sample.numel()
    total_size_bytes = sample_size_bytes * len(dataset)
    return total_size_bytes / (1024**3)


# Backward-compatible alias for older notebook code.
estimate_dataset_size = estimate_dataset_size_gb


def get_label_distribution(dataset) -> Counter:
    """Return counts for labels in a PyTorch-style dataset."""
    return Counter(label for _, label in dataset)


def load_and_average_frame_intensity(
    directory: str | Path,
    expected_shape: tuple[int, ...] | None = (1, 180, 500, 500),
) -> np.ndarray:
    """Compute mean frame intensity across all `.pt` tensor files in a directory.

    Parameters
    ----------
    directory:
        Folder containing `.pt` tensors.
    expected_shape:
        Optional tensor shape check. Use `None` to disable shape validation.

    Returns
    -------
    np.ndarray
        One-dimensional array where each value is the average intensity for a
        frame, averaged across all tensor files.
    """
    directory = Path(directory)
    pt_files = sorted(directory.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {directory}")

    time_series: list[np.ndarray] = []
    for file_path in pt_files:
        data = torch.load(file_path, map_location="cpu")

        if expected_shape is not None and tuple(data.shape) != expected_shape:
            raise ValueError(
                f"Unexpected shape {tuple(data.shape)} in {file_path.name}; "
                f"expected {expected_shape}."
            )

        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)  # [T, H, W]
        elif data.dim() != 3:
            raise ValueError(
                f"Expected tensor with shape [1, T, H, W] or [T, H, W], "
                f"got {tuple(data.shape)} in {file_path.name}."
            )

        avg_brightness = data.float().mean(dim=(1, 2))
        time_series.append(avg_brightness.numpy())

    return np.mean(np.stack(time_series, axis=0), axis=0)


# Backward-compatible alias for older notebook code.
load_and_average_images = load_and_average_frame_intensity


def export_frame_intensity_to_csv(
    frame_intensity: Sequence[float] | np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write frame-level average intensity values to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Frame": np.arange(len(frame_intensity)),
            "Average Intensity": np.asarray(frame_intensity),
        }
    )
    df.to_csv(output_path, index=False)
    return output_path


# Backward-compatible alias for older notebook code.
export_to_csv = export_frame_intensity_to_csv


def reduce_tensor_to_7_frames(
    tensor_data: torch.Tensor,
    selected_frame_indices: Sequence[int] = DEFAULT_SELECTED_FRAME_INDICES,
    baseline_frame_count: int = 20,
    target_size: tuple[int, int] = (500, 500),
) -> torch.Tensor:
    """Reduce a full time-series tensor to the 7-frame ANSA model input.

    The output contains one baseline frame formed by averaging the first
    `baseline_frame_count` frames, followed by frames at `selected_frame_indices`.

    Expected input shape is `[C, T, H, W]`. Output shape is `[1, 7, H, W]` after
    resizing to `target_size`.
    """
    if tensor_data.dim() != 4:
        raise ValueError(
            f"Expected tensor with shape [C, T, H, W], got {tuple(tensor_data.shape)}."
        )

    max_frames = tensor_data.shape[1]
    if max_frames < baseline_frame_count:
        raise ValueError(
            f"Tensor has {max_frames} frames, fewer than baseline_frame_count="
            f"{baseline_frame_count}."
        )

    missing_frames = [idx for idx in selected_frame_indices if idx >= max_frames]
    if missing_frames:
        raise ValueError(
            f"Tensor has {max_frames} frames and is missing selected frame indices: "
            f"{missing_frames}."
        )

    avg_baseline = tensor_data[:, :baseline_frame_count, :, :].float().mean(
        dim=1, keepdim=True
    )
    selected_frames = tensor_data[:, list(selected_frame_indices), :, :].float()
    reduced = torch.cat((avg_baseline, selected_frames), dim=1)  # [C, 7, H, W]

    if reduced.shape[0] != 1:
        raise ValueError(
            f"Expected single-channel tensor after reduction, got {reduced.shape[0]} channels."
        )

    return F.interpolate(reduced, size=target_size, mode="bilinear", align_corners=False)


def reduce_dataset_to_7_frame_tensors(
    root_dir: str | Path,
    save_dir: str | Path,
    class_names: Iterable[str] = DEFAULT_CLASS_NAMES,
    selected_frame_indices: Sequence[int] = DEFAULT_SELECTED_FRAME_INDICES,
    baseline_frame_count: int = 20,
    target_size: tuple[int, int] = (500, 500),
    overwrite: bool = False,
) -> list[Path]:
    """Reduce all `.pt` tensors in class folders to 7-frame tensors.

    Parameters
    ----------
    root_dir:
        Input directory containing one folder per class.
    save_dir:
        Output directory where reduced tensors are written using the same class
        folder structure.
    class_names:
        Class folder names to process.
    selected_frame_indices:
        Frame indices to append after the averaged baseline frame.
    baseline_frame_count:
        Number of initial frames to average into the baseline frame.
    target_size:
        Spatial output size as `(height, width)`.
    overwrite:
        Whether to overwrite existing output `.pt` files.

    Returns
    -------
    list[pathlib.Path]
        Paths to files written.
    """
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    written_files: list[Path] = []

    for class_name in class_names:
        class_path = root_dir / class_name
        save_class_path = save_dir / class_name

        if not class_path.exists():
            continue

        save_class_path.mkdir(parents=True, exist_ok=True)

        for file_path in sorted(class_path.glob("*.pt")):
            save_file_path = save_class_path / file_path.name
            if save_file_path.exists() and not overwrite:
                continue

            tensor_data = torch.load(file_path, map_location="cpu")
            reduced_tensor = reduce_tensor_to_7_frames(
                tensor_data=tensor_data,
                selected_frame_indices=selected_frame_indices,
                baseline_frame_count=baseline_frame_count,
                target_size=target_size,
            )
            torch.save(reduced_tensor, save_file_path)
            written_files.append(save_file_path)

    return written_files


# Backward-compatible alias for older notebook code.
process_and_save = reduce_dataset_to_7_frame_tensors
