"""Profiles -- the per-pipeline bundle of defaults.

A :class:`Profile` is the answer to "which pipeline is this?". Everything that
differs between the cerebellar and cortical pipelines is a *value* here, not a
branch somewhere in the code: which ROI atlas, which time windows, whether to
mask, whether to regress the global signal.

The stage chain itself is fixed::

    correction -> [mask] -> [GSR] -> ROI -> connectivity

with the bracketed stages switched on by the profile. Both pipelines are that
same chain; the cerebellar one simply leaves the brackets empty.

**Presets, not a closed set.** The three below are conveniences. A study builds
its own -- from scratch or with ``dataclasses.replace(CORTICAL_GSR, trim=5)`` --
and never edits this file. Profiles carry *pipeline* differences only; anything
about a cohort, a group or an animal belongs in a study script, not here
(MERGING_PLAN.md P2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .atlases import CEREBELLUM_4, CORTEX_22, Atlas, as_atlas
from .gsr import GSRConfig

# Channel order in an interleaved folder. "auto" identifies the channels by
# brightness (the DIMMER group is GCaMP; the reflectance/emo channel comes back
# brighter on this rig), which is more robust than position -- the MATLAB assigned
# channels purely by their position in the `dir` listing, which is exactly the
# silent channel-swap risk the Python side set out to remove. The positional
# options exist to override the heuristic when it is wrong (e.g. an unusually
# bright GCaMP recording).
CHANNEL_ORDERS = ("auto", "gcamp_first", "emo_first")


@dataclass(frozen=True)
class Profile:
    """A named bundle of pipeline settings.

    Attributes
    ----------
    name:
        Identifier, used in reports and by the CLI's ``--profile``.
    atlas:
        An :class:`~wfci.atlases.Atlas`, or any ``{label: Box}`` mapping in column
        order (wrapped into an Atlas with no declared grid). See
        :mod:`wfci.atlases`.
    trim:
        Frames dropped from the front of every trial. 20 for the cerebellar
        scripts (``out(:,:,21:end)``); 0 for the cortical ones, which do not trim.
    baseline:
        Temporal window for the Delta F/F baseline image (step 1).
    corr_window:
        Temporal window the correlation is computed over (step 3).
    use_mask:
        Whether the pipeline expects a brain mask. When True a mask must actually
        be supplied -- it is not optional-in-practice, so the run fails rather
        than quietly analysing the whole FOV including skull and background.
    gsr:
        :class:`~wfci.gsr.GSRConfig` to regress out the global signal, or None.
    mask_downsample:
        Scale applied to the mask before use, so it lands on the same grid as the
        twice-downsampled data. The cortical mask is drawn on the once-downsampled
        FOV, hence 0.5 -- matching ``imresize(Mask,0.5,'box')`` in script (2). Use
        ``None`` for a mask already at the final resolution.
    overlay_frame:
        Default frame index for the step-2 ROI-placement figure.
    channel_order:
        See :data:`CHANNEL_ORDERS`.
    """

    name: str
    atlas: Atlas | Mapping
    trim: int = 20
    # default_factory, not a plain default: `slice` is unhashable on Python 3.11
    # and dataclasses reject unhashable defaults as mutable. (Harmless here --
    # slices are immutable in practice -- but 3.11 is this project's floor.)
    baseline: slice = field(default_factory=lambda: slice(None))
    corr_window: slice = field(default_factory=lambda: slice(None))
    use_mask: bool = False
    gsr: GSRConfig | None = None
    mask_downsample: float | None = 0.5
    overlay_frame: int = 302
    channel_order: str = "auto"
    downsample: float = 0.5

    def __post_init__(self) -> None:
        if self.channel_order not in CHANNEL_ORDERS:
            raise ValueError(
                f"channel_order={self.channel_order!r} is not one of {CHANNEL_ORDERS}."
            )
        if not self.atlas:
            raise ValueError(f"Profile {self.name!r} has an empty atlas.")
        # Normalise to an Atlas so `profile.atlas.grid` always exists (a plain dict
        # gets grid=None, i.e. "unknown, do not check"). Atlas copies its boxes, so
        # this also stops a study mutating its profile's atlas from silently
        # rewriting the shared CEREBELLUM_4 / CORTEX_22 presets for the whole
        # process -- `frozen=True` only stops the field being reassigned, not the
        # mapping behind it being edited.
        object.__setattr__(self, "atlas", as_atlas(self.atlas, name=f"{self.name}_atlas"))

    @property
    def labels(self) -> list[str]:
        """ROI labels in column order (i.e. the atlas's own order)."""
        return list(self.atlas.keys())

    @property
    def n_rois(self) -> int:
        return len(self.atlas)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# Cerebellar, resting state: baseline and correlation both span the whole
# recording. This is what run_resting_state() has always done -- and it is
# MATLAB-validated to machine precision, so these values are frozen by P1.
CEREBELLAR_RS = Profile(
    name="cerebellar_rs",
    atlas=CEREBELLUM_4,
    trim=20,
    baseline=slice(None),
    corr_window=slice(None),
    overlay_frame=302,
)

# Cerebellar, stimulated: pre-stimulus baseline (MATLAB 1:278) and a correlation
# window over the stimulus response (MATLAB 280:300).
CEREBELLAR_STIM = Profile(
    name="cerebellar_stim",
    atlas=CEREBELLUM_4,
    trim=20,
    baseline=slice(0, 278),
    corr_window=slice(279, 300),
    overlay_frame=302,
)

# Cortical with GSR (the Antea scripts). No trim -- script (1) keeps every frame;
# full-recording baseline; brain mask + global signal regression before the ROI
# means. overlay_frame=10 mirrors script (3)'s t_TEMP_regressed(:,:,11,2).
#
# NOTE (MERGING_PLAN.md P4): unlike the cerebellar presets, these settings have no
# MATLAB reference behind them -- they are a careful reading of the scripts, not a
# proven match.
CORTICAL_GSR = Profile(
    name="cortical_gsr",
    atlas=CORTEX_22,
    trim=0,
    baseline=slice(None),
    corr_window=slice(None),
    use_mask=True,
    gsr=GSRConfig(),
    mask_downsample=0.5,
    overlay_frame=10,
)

PROFILES: dict[str, Profile] = {
    p.name: p for p in (CEREBELLAR_RS, CEREBELLAR_STIM, CORTICAL_GSR)
}


def get_profile(name: str) -> Profile:
    """Look up a preset by name, with a useful error for a typo."""
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown profile {name!r}. Available: {sorted(PROFILES)}. "
            f"Profiles are presets, not a closed set -- build your own Profile "
            f"(or dataclasses.replace one of these) for a study-specific setup."
        ) from None
