"""wfci - Wide-field Calcium Imaging ROI functional connectivity.

Python port of the MATLAB wide-field pipelines (hemodynamic correction, ROI
placement, region-to-region functional connectivity) for dual-channel imaging of
mouse brain, for resting-state and stimulated recordings.

The package is anatomy-agnostic: the ROI layout is an argument (see
:mod:`wfci.atlases`), not a structural assumption. It ships two presets --
``CEREBELLUM_4`` (the cerebellar pipeline, validated against MATLAB to machine
precision) and ``CORTEX_22`` (the cortical pipeline) -- and any other dict of
boxes works just as well.
"""

from __future__ import annotations

from .atlases import (
    ATLASES,
    CEREBELLUM_4,
    CORTEX_22,
    Atlas,
    as_atlas,
    atlas_labels,
    load_atlas,
    save_atlas,
)
from .cohort import (
    AnimalResult,
    CohortTable,
    LabeledMatrix,
    load_results,
)
from .config import Box, ROIConfig
from .correction import build_dff_stack, hemodynamic_correction
from .dump import VOLUME_NAMES, PixelDump
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
from .gsr import GSRConfig, global_signal, regress_global
from .mask import apply_mask, load_mask, resize_mask, valid_from_mask
from .pipeline import (
    PipelineResult,
    run_pipeline,
    run_profile,
    run_resting_state,
    run_stimulated,
)
from .profiles import (
    CEREBELLAR_RS,
    CEREBELLAR_STIM,
    CORTICAL_GSR,
    PROFILES,
    Profile,
    get_profile,
)
from .resize import imresize_box
from .significance import (
    circular_layout,
    count_significant_edges,
    hemispheric_layout,
    mask_by_adjacency,
    network_figure,
    node_strength,
    significance_barplot,
)
from .roi import box_slices_for, extract_roi_timeseries, functional_connectivity
from .streaming import (
    StreamingResult,
    run_streaming,
    run_streaming_profile,
    run_streaming_resting_state,
    run_streaming_stimulated,
    stream_trial_roi,
)
from .visualize import overlay_rois, show_roi_placement

__all__ = [
    "Box",
    "ROIConfig",
    "Atlas",
    "CEREBELLUM_4",
    "CORTEX_22",
    "ATLASES",
    "atlas_labels",
    "as_atlas",
    "load_atlas",
    "save_atlas",
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
    "load_mask",
    "resize_mask",
    "valid_from_mask",
    "apply_mask",
    "hemodynamic_correction",
    "build_dff_stack",
    "GSRConfig",
    "global_signal",
    "regress_global",
    "extract_roi_timeseries",
    "box_slices_for",
    "functional_connectivity",
    "Profile",
    "PROFILES",
    "get_profile",
    "CEREBELLAR_RS",
    "CEREBELLAR_STIM",
    "CORTICAL_GSR",
    "run_pipeline",
    "run_profile",
    "run_resting_state",
    "run_stimulated",
    "PipelineResult",
    "stream_trial_roi",
    "run_streaming",
    "run_streaming_profile",
    "run_streaming_resting_state",
    "run_streaming_stimulated",
    "StreamingResult",
    "PixelDump",
    "VOLUME_NAMES",
    "overlay_rois",
    "show_roi_placement",
    "AnimalResult",
    "CohortTable",
    "LabeledMatrix",
    "load_results",
    "mask_by_adjacency",
    "node_strength",
    "count_significant_edges",
    "circular_layout",
    "hemispheric_layout",
    "network_figure",
    "significance_barplot",
]

__version__ = "0.1.0"
