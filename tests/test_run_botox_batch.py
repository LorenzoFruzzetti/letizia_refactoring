from __future__ import annotations

import argparse
import os

import numpy as np
import pytest

import run_botox_batch as batch


def test_resolve_workers_uses_75_percent_and_caps_explicit_values(monkeypatch):
    monkeypatch.setattr(batch.os, 'cpu_count', lambda: 20)

    assert batch.resolve_workers(None) == 15
    assert batch.resolve_workers(0) == 15
    assert batch.resolve_workers(1) == 1
    assert batch.resolve_workers(99) == 20


def test_project_default_is_reliable_serial_execution():
    args = batch.build_runtime_args(dict(batch.RUN_CONFIG, prefer_cli_args=False))

    assert args.workers == 1
    assert batch.resolve_workers(args.workers) == 1


def test_summary_from_npz_recovers_completed_job(tmp_path):
    out_dir = tmp_path / '260611_PV5' / 't1'
    out_dir.mkdir(parents=True)
    np.savez(
        out_dir / 'connectivity_full.npz',
        day=np.array('260611'),
        animal=np.array('PV5'),
        group=np.array('P'),
        profile=np.array('cerebellar_rs'),
        bregma=np.array([90, 130]),
        roi_set=np.array(r'roi_sets\rebuilt\260611_PV5_t1.yaml'),
        R_mean=np.array([[1.0, 0.25], [0.25, 1.0]]),
    )
    args = argparse.Namespace(debug=False)
    job = {
        'args': args,
        'row': {'day': '260611', 'animal': 'PV5'},
        'label': 't1',
        'folders': ['recording-folder'],
        'out_dir': str(out_dir),
    }

    row = batch._summary_from_npz(job)

    assert batch._summary_key(row) == ('260611', 'PV5', 'full', 't1')
    assert row['bregma_row'] == 90
    assert row['bregma_col'] == 130
    assert row['mean_offdiag_R'] == 0.25
    assert row['elapsed_s'] == ''


def test_keyboard_interrupt_terminates_pool_workers(tmp_path, monkeypatch):
    args = argparse.Namespace(
        manifest='unused.csv', output_root=str(tmp_path), select=[], max_rows=None,
        roi_set_dir=None, roi_set=None, debug=False, debug_frames=60,
        max_recordings=None, merge_recordings=False, workers=2,
        skip_existing=False,
    )
    rows = [{
        'day': '260611', 'animal': 'PV5', 'group': 'P',
        'profile': 'cerebellar_rs', 'bregma_row': '90', 'bregma_col': '130',
        'channel_order': 'auto', 'recording_paths': r'input\t1',
    }]

    class FakeFuture:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    class FakeProcess:
        terminated = False
        joined = False

        def is_alive(self):
            return True

        def terminate(self):
            self.terminated = True

        def join(self, timeout):
            assert timeout == 5
            self.joined = True

    future = FakeFuture()
    process = FakeProcess()

    class FakePool:
        def __init__(self, max_workers):
            assert max_workers == 2
            self._processes = {1: process}
            self.shutdown_args = None

        def submit(self, function, job):
            assert function is batch.run_job
            return future

        def shutdown(self, **kwargs):
            self.shutdown_args = kwargs

    pool = FakePool(2)
    monkeypatch.setattr(batch, 'build_runtime_args', lambda: args)
    monkeypatch.setattr(batch, 'load_manifest', lambda _path: rows)
    monkeypatch.setattr(batch, 'ProcessPoolExecutor', lambda max_workers: pool)
    monkeypatch.setattr(batch, 'wait', lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))

    with pytest.raises(SystemExit) as stopped:
        batch.main()

    assert stopped.value.code == 130
    assert future.cancelled
    assert process.terminated
    assert process.joined
    assert pool.shutdown_args == {'wait': False, 'cancel_futures': True}


def test_workers_one_never_constructs_a_process_pool(tmp_path, monkeypatch):
    args = argparse.Namespace(
        manifest='unused.csv', output_root=str(tmp_path), select=[], max_rows=None,
        roi_set_dir=None, roi_set=None, debug=False, debug_frames=60,
        max_recordings=None, merge_recordings=False, workers=1,
        skip_existing=False,
    )
    rows = [{
        'day': '260611', 'animal': 'PV5', 'group': 'P',
        'profile': 'cerebellar_rs', 'bregma_row': '90', 'bregma_col': '130',
        'channel_order': 'auto', 'recording_paths': r'input\t1',
    }]
    summary = dict(
        day='260611', animal='PV5', group='P', profile='cerebellar_rs',
        bregma_row=90, bregma_col=130, roi_set='', mode='full', recording='t1',
        n_recordings=1, mean_offdiag_R=0.25, elapsed_s=1.0, npz_path='result.npz',
    )
    executed = []

    monkeypatch.setattr(batch, 'build_runtime_args', lambda: args)
    monkeypatch.setattr(batch, 'load_manifest', lambda _path: rows)
    monkeypatch.setattr(
        batch, 'ProcessPoolExecutor',
        lambda *_args, **_kwargs: pytest.fail('serial mode constructed a process pool'),
    )
    monkeypatch.setattr(
        batch, 'execute_job',
        lambda job: (executed.append(job['label']) or summary),
    )

    batch.main()

    assert executed == ['t1']
    recovered = batch._read_summary_rows(str(tmp_path / 'batch_summary.csv'))
    assert len(recovered) == 1
    assert batch._summary_key(recovered[0]) == ('260611', 'PV5', 'full', 't1')


def test_save_data_is_off_by_default_and_reaches_the_namespace():
    """Every save_data key must survive build_runtime_args' editor path.

    That path builds an explicit Namespace field by field rather than splatting
    RUN_CONFIG, so a key added to the dict but not to the literal is not caught
    until a run reaches the code that reads it -- late, and only when the flag is
    on. Assert the whole set instead.
    """
    args = batch.build_runtime_args(dict(batch.RUN_CONFIG, prefer_cli_args=False))

    assert args.save_data is False, "save_data must default to off"
    assert args.save_data_root is None
    assert args.save_data_window == (-29, 47, -44, 43)
    assert args.save_data_dff_dtype == 'float16'
    assert args.save_data_f_dtype == 'float32'


def test_dump_region_is_anchored_to_this_recordings_bregma():
    """The window is Bregma-relative, so two animals get different rectangles.

    That is the property that makes dumps comparable across animals: the same
    window names the same anatomy, not the same camera pixels.
    """
    from wfci.config import Box

    args = argparse.Namespace(save_data_window=(-2, 3, -2, 3))
    boxes = {'a': Box(-1, 2, -1, 2)}

    from wfci import ROIConfig

    assert batch._dump_region(ROIConfig(y_1=10, x_2=20, boxes=boxes), args) == (8, 13, 18, 23)
    assert batch._dump_region(ROIConfig(y_1=30, x_2=40, boxes=boxes), args) == (28, 33, 38, 43)


def test_dump_region_refuses_a_box_outside_the_window():
    """A clipped ROI box would make the dump quietly disagree with the analysis."""
    from wfci import ROIConfig
    from wfci.config import Box

    args = argparse.Namespace(save_data_window=(-2, 3, -2, 3))
    cfg = ROIConfig(y_1=10, x_2=20, boxes={'outside': Box(1, 9, 1, 2)})

    with pytest.raises(ValueError, match='does not contain these ROI boxes'):
        batch._dump_region(cfg, args)


def test_configured_window_contains_every_rebuilt_roi_set():
    """The shipped window is the measured union of all roi_sets\rebuilt boxes.

    It is a constant in RUN_CONFIG, so nothing stops a future edit to those sets
    from outgrowing it. This test makes that show up here rather than as a
    refusal partway through a batch.
    """
    import glob
    import os

    import yaml

    from wfci import ROIConfig
    from wfci.config import Box

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(batch.__file__))),
                        'letizia', 'roi_sets', 'rebuilt')
    if not os.path.isdir(root):
        root = os.path.join(os.path.dirname(os.path.abspath(batch.__file__)),
                            'roi_sets', 'rebuilt')
    paths = sorted(glob.glob(os.path.join(root, '*.yaml')))
    if not paths:
        pytest.skip('roi_sets/rebuilt is not present in this checkout')

    args = argparse.Namespace(save_data_window=batch.RUN_CONFIG['save_data_window'])
    grid = 128
    for path in paths:
        with open(path) as fh:
            doc = yaml.safe_load(fh)
        cfg = ROIConfig(y_1=doc['bregma_row'] // 2, x_2=doc['bregma_col'] // 2,
                        boxes={k: Box(**v) for k, v in doc['boxes'].items()})
        region = batch._dump_region(cfg, args)   # raises if a box falls outside
        r0, r1, c0, c1 = region
        assert 0 <= r0 < r1 <= grid and 0 <= c0 < c1 <= grid, (
            f'{os.path.basename(path)}: window {region} leaves the {grid}x{grid} grid'
        )


def _job(tmp_path, out_dir):
    return dict(row={'day': '260611', 'animal': 'PV5'}, args=None,
                folders=['x'], rec_names=['t1'], label='t1', out_dir=str(out_dir))


def test_resume_requires_the_dump_when_save_data_is_on(tmp_path):
    """A finished analysis is NOT a finished job once --save-data is added.

    Every recording in this study already has its connectivity_full.npz, so
    resuming on that marker alone would skip all 355 jobs and write no pixels --
    the flag would look like it ran and produce nothing.
    """
    out_dir = tmp_path / 'out' / '260611_PV5' / 't1'
    out_dir.mkdir(parents=True)
    (out_dir / 'connectivity_full.npz').write_bytes(b'')
    job = _job(tmp_path, out_dir)

    off = argparse.Namespace(debug=False, save_data=False, output_root=str(tmp_path / 'out'),
                             save_data_root=None)
    on = argparse.Namespace(debug=False, save_data=True, output_root=str(tmp_path / 'out'),
                            save_data_root=None)

    assert batch._job_is_complete(job, off) is True, 'the .npz alone completes a no-dump job'
    assert batch._job_is_complete(job, on) is False, 'a missing dump must not count as done'

    (out_dir / 'pixels_meta_full.npz').write_bytes(b'')
    assert batch._job_is_complete(job, on) is True, 'the dump sidecar completes it'


def test_resume_looks_for_the_dump_under_save_data_root(tmp_path):
    """With the volumes on another disk, resume must look for them there."""
    out_root = tmp_path / 'out'
    dump_root = tmp_path / 'pixels'
    out_dir = out_root / '260611_PV5' / 't1'
    out_dir.mkdir(parents=True)
    (out_dir / 'connectivity_full.npz').write_bytes(b'')
    job = _job(tmp_path, out_dir)
    args = argparse.Namespace(debug=False, save_data=True, output_root=str(out_root),
                              save_data_root=str(dump_root))

    assert batch._job_is_complete(job, args) is False

    mirrored = dump_root / '260611_PV5' / 't1'
    mirrored.mkdir(parents=True)
    (mirrored / 'pixels_meta_full.npz').write_bytes(b'')
    assert batch._job_is_complete(job, args) is True


def test_save_data_dtypes_and_root_are_settable_from_the_cli(monkeypatch):
    """The output path and both storage dtypes must be reachable without editing code."""
    monkeypatch.setattr(
        batch.sys, 'argv',
        ['run_botox_batch.py', '--save-data', '--save-data-root', r'D:\pixels',
         '--save-data-dff-dtype', 'float32', '--save-data-f-dtype', 'float16'],
    )
    args = batch.build_runtime_args(dict(batch.RUN_CONFIG))

    assert args.save_data is True
    assert args.save_data_root == r'D:\pixels'
    assert args.save_data_dff_dtype == 'float32'
    assert args.save_data_f_dtype == 'float16'
    # Not a flag: it changes which pixels the array contains, not just where it goes.
    assert args.save_data_window == batch.RUN_CONFIG['save_data_window']


def test_save_data_root_mirrors_the_output_tree(tmp_path):
    """Volumes land under save_data_root at the same <day>_<animal>\<t#> tail.

    That is what lets ~70 GB of pixels sit on a different disk from the small
    analysis outputs while staying matched up recording by recording.
    """
    args = argparse.Namespace(output_root=r'outputs\run', save_data_root=r'D:\pixels')
    out_dir = os.path.join(r'outputs\run', '260611_PV5', 't1')

    assert batch._dump_dir(out_dir, args) == os.path.join(r'D:\pixels', '260611_PV5', 't1')

    beside = argparse.Namespace(output_root=r'outputs\run', save_data_root=None)
    assert batch._dump_dir(out_dir, beside) == out_dir
