"""Visualization helpers for ANSA dataset inspection.

This module contains plotting functions extracted from the original
`Accessory_functions.py`. Functions return matplotlib axes so notebooks can
customize or save figures after plotting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LABEL_MAPPING: dict[int, str] = {
    0: "Und.",
    1: "Low",
    2: "Med",
    3: "High",
    4: "Very high",
}


def plot_label_distribution(
    label_counts: Mapping[int, int],
    title: str = "Label distribution",
    label_mapping: Mapping[int, str] | None = None,
    ax=None,
):
    """Plot class counts from a label-count mapping.

    Parameters
    ----------
    label_counts:
        Mapping from integer label to count, such as the output of
        `get_label_distribution`.
    title:
        Plot title.
    label_mapping:
        Optional mapping from integer labels to display labels.
    ax:
        Optional matplotlib axes. If omitted, a new figure and axes are created.
    """
    label_mapping = label_mapping or DEFAULT_LABEL_MAPPING
    labels, counts = zip(*sorted(label_counts.items())) if label_counts else ([], [])
    display_labels = [label_mapping.get(label, str(label)) for label in labels]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.bar(display_labels, counts)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax


def plot_frame_intensity(
    frame_intensity: Sequence[float] | np.ndarray | pd.Series,
    title: str = "Brightness change over time",
    ax=None,
):
    """Plot average frame intensity across time."""
    values = np.asarray(frame_intensity)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(np.arange(len(values)), values, marker="o", linestyle="-", label="Avg brightness")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Average brightness")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax


# Backward-compatible alias for older notebook code.
plot_brightness = plot_frame_intensity


def plot_frame_selection_overlay(
    frame_intensity: Sequence[float] | np.ndarray | pd.Series,
    frame_sets: Mapping[str, Sequence[int]],
    title: str = "Selected frames over average intensity",
    ax=None,
):
    """Plot average frame intensity with vertical markers for selected frames.

    Parameters
    ----------
    frame_intensity:
        One-dimensional frame intensity values.
    frame_sets:
        Mapping from legend label to a list of frame indices.
    title:
        Plot title.
    ax:
        Optional matplotlib axes. If omitted, a new figure and axes are created.
    """
    values = np.asarray(frame_intensity)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(np.arange(len(values)), values, label="Average intensity")
    ymax = float(np.nanmax(values)) if values.size else 1.0

    for label, frames in frame_sets.items():
        ax.vlines(frames, ymin=0, ymax=ymax, linestyles="dashed", alpha=0.8, label=label)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Intensity average over all images")
    ax.set_title(title)
    ax.legend()
    return ax
