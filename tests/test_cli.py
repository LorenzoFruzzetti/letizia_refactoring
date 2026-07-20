"""The CLI contract: profile selection, the deprecated --mode, and mask rules.

run_pipeline.py is the entrypoint people actually type, so its argument handling
is behaviour like any other. Two things matter most here:

  * ``--mode`` must keep working exactly as it did (MERGING_PLAN.md P1 -- no
    documented command may change), while ``--profile`` supersedes it;
  * the mask rules must be enforced *before* a long run, not discovered after it:
    a cortical run without a mask would silently average skull and background
    into the global signal, which is a wrong result rather than a crash.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from run_pipeline import (  # noqa: E402
    MODE_TO_PROFILE,
    PROFILE_CHOICES,
    _resolve_profile_name,
    build_runtime_args,
)
from wfci import PROFILES, get_profile  # noqa: E402

DEFAULTS = {"profile": "cerebellar_rs"}


# ---------------------------------------------------------------------------
# --profile / --mode
# ---------------------------------------------------------------------------
def test_every_profile_preset_is_offered_by_the_cli():
    """The CLI's choices are derived from the presets, never a stale second list."""
    assert PROFILE_CHOICES == sorted(PROFILES)
    assert set(PROFILE_CHOICES) == {"cerebellar_rs", "cerebellar_stim", "cortical_gsr"}


def test_mode_maps_onto_the_cerebellar_profiles():
    """--mode only ever named a cerebellar pipeline; that mapping is the alias."""
    assert _resolve_profile_name("cerebellar_rs", "resting_state", DEFAULTS) == "cerebellar_rs"
    assert _resolve_profile_name("cerebellar_rs", "stimulated", DEFAULTS) == "cerebellar_stim"


def test_mode_is_the_only_deprecated_spelling_needed():
    assert set(MODE_TO_PROFILE) == {"resting_state", "stimulated"}


def test_profile_alone_is_untouched_by_the_alias():
    assert _resolve_profile_name("cortical_gsr", None, DEFAULTS) == "cortical_gsr"


def test_mode_and_a_contradicting_profile_are_refused():
    """Ambiguity gets an error, not a silent winner.

    Guessing here would mean running a different pipeline than the user typed.
    """
    with pytest.raises(SystemExit, match="Pass only --profile"):
        _resolve_profile_name("cortical_gsr", "resting_state", DEFAULTS)


def test_mode_agreeing_with_an_explicit_profile_is_accepted():
    """Redundant but consistent is not an error."""
    assert _resolve_profile_name("cerebellar_stim", "stimulated", DEFAULTS) == "cerebellar_stim"


# ---------------------------------------------------------------------------
# RUN_CONFIG (the editor path)
# ---------------------------------------------------------------------------
def _config(**overrides) -> dict:
    cfg = {
        "trials": ["some/folder"],
        "source": "interleaved_folder",
        "streaming": True,
        "profile": "cerebellar_rs",
        "mask_path": None,
        "channel_order": "auto",
        "bregma_row": 121,
        "bregma_col": 134,
        "output_path": None,
        "debug_max_frames": None,
        "prefer_cli_args": False,
    }
    cfg.update(overrides)
    return cfg


def test_run_config_exposes_the_new_knobs():
    args = build_runtime_args(_config(profile="cortical_gsr", mask_path="m.tif",
                                     channel_order="emo_first"))

    assert args.profile == "cortical_gsr"
    assert args.mask == "m.tif"
    assert args.channel_order == "emo_first"


def test_a_pre_profiles_run_config_still_works():
    """An editor setup written before profiles existed used a "mode" key.

    P1 again: someone's RUN_CONFIG on disk must not need editing to keep running.
    """
    old_style = _config()
    del old_style["profile"]
    old_style["mode"] = "stimulated"

    args = build_runtime_args(old_style)

    assert args.profile == "cerebellar_stim"


def test_run_config_defaults_channel_order_when_absent():
    cfg = _config()
    del cfg["channel_order"]

    assert build_runtime_args(cfg).channel_order == "auto"


# ---------------------------------------------------------------------------
# Profile lookup
# ---------------------------------------------------------------------------
def test_an_unknown_profile_name_says_what_is_available():
    with pytest.raises(ValueError, match="Unknown profile"):
        get_profile("cortical")  # a plausible near-miss for "cortical_gsr"


def test_the_profiles_describe_the_two_pipelines():
    """Pin the preset values: these are the pipeline definitions themselves."""
    rs = get_profile("cerebellar_rs")
    assert (rs.trim, rs.n_rois, rs.use_mask, rs.gsr) == (20, 4, False, None)
    assert rs.baseline == slice(None) and rs.corr_window == slice(None)

    stim = get_profile("cerebellar_stim")
    assert (stim.trim, stim.n_rois, stim.use_mask, stim.gsr) == (20, 4, False, None)
    assert stim.baseline == slice(0, 278)      # MATLAB 1:278
    assert stim.corr_window == slice(279, 300)  # MATLAB 280:300

    cortical = get_profile("cortical_gsr")
    assert (cortical.trim, cortical.n_rois, cortical.use_mask) == (0, 22, True)
    assert cortical.gsr is not None
    assert cortical.mask_downsample == 0.5


def test_a_profile_copies_the_atlas_it_is_given():
    """A profile must not alias the caller's (or the preset's) atlas dict.

    ``frozen=True`` stops the *field* being reassigned; it does nothing about the
    dict being mutated in place. Without a copy, a study that tweaked its own
    profile's atlas would silently rewrite CORTEX_22 for every other profile in
    the process -- including presets it never touched.
    """
    from wfci.profiles import Profile
    from wfci.atlases import CORTEX_22

    mine = dict(CORTEX_22)
    p = Profile(name="mine", atlas=mine)

    mine.pop("V1R")  # mutate the dict we handed in

    assert "V1R" in p.atlas, "the profile aliased the caller's dict"
    assert "V1R" in CORTEX_22, "the shared preset was mutated"


def test_the_cortical_preset_still_holds_the_full_atlas():
    """The presets are module-level singletons; nothing may have eaten one."""
    from wfci.atlases import CORTEX_22

    assert get_profile("cortical_gsr").atlas == CORTEX_22
    assert len(CORTEX_22) == 22


def test_an_invalid_channel_order_is_refused():
    from wfci.profiles import Profile
    from wfci.atlases import CEREBELLUM_4

    with pytest.raises(ValueError, match="channel_order"):
        Profile(name="x", atlas=CEREBELLUM_4, channel_order="gcamp")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
