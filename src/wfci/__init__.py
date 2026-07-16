"""wfci - Wide-field Calcium Imaging cerebellar ROI functional connectivity.

Python port of the MATLAB pipeline (hemodynamic correction, ROI placement,
region-to-region functional connectivity) for dual-channel wide-field imaging
of mouse cerebellum, for resting-state and stimulated recordings.
"""

from __future__ import annotations

from .config import Box, ROIConfig
from .correction import build_dff_stack, hemodynamic_correction
from .io import (
    FrameSource,
    folder_frame_source,
    frame_folder_source,
    interleaved_channel_files,
    iter_folder_frames,
    iter_tiff_frames,
    load_frame_folder,
    load_interleaved_folder,
    load_stack,
    tiff_frame_count,
    tiff_frame_source,
)
from .pipeline import PipelineResult, run_pipeline, run_resting_state, run_stimulated
from .resize import imresize_box
from .roi import extract_roi_timeseries, functional_connectivity
from .streaming import (
    StreamingResult,
    run_streaming,
    run_streaming_resting_state,
    run_streaming_stimulated,
    stream_trial_roi,
)
from .visualize import overlay_rois, show_roi_placement

__all__ = [
    "Box",
    "ROIConfig",
    "load_stack",
    "load_frame_folder",
    "load_interleaved_folder",
    "interleaved_channel_files",
    "iter_tiff_frames",
    "iter_folder_frames",
    "tiff_frame_count",
    "FrameSource",
    "tiff_frame_source",
    "folder_frame_source",
    "frame_folder_source",
    "imresize_box",
    "hemodynamic_correction",
    "build_dff_stack",
    "extract_roi_timeseries",
    "functional_connectivity",
    "run_pipeline",
    "run_resting_state",
    "run_stimulated",
    "PipelineResult",
    "stream_trial_roi",
    "run_streaming",
    "run_streaming_resting_state",
    "run_streaming_stimulated",
    "StreamingResult",
    "overlay_rois",
    "show_roi_placement",
]

__version__ = "0.1.0"
