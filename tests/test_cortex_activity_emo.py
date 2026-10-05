"""Prespecified synthetic checks for additive correction and valid exposure."""
import numpy as np
import pandas as pd
import pytest
from analyze_cortex_activity_emo import RUN_CONFIG, correct_signal, detect_events, metric_row
from compare_cortex_activity_conditions import aggregate_exposure


def test_additive_decline_pulses_and_offset_invariance():
    n,fs = 3000,10
    t = np.arange(n)/fs
    pulses = np.zeros(n)
    for a in (700,1400,2100):
        pulses[a:a+10] = .02
    results = []
    for slope in (0,.00002,.0001):
        q = (1-slope*t+pulses)[:,None]
        r = correct_signal(q,601)
        np.testing.assert_allclose(r["corrected"]-r["mean_b"],r["residual"],atol=2e-16)
        np.testing.assert_allclose(r["residual"][[705,1405,2105],0],.02,atol=.00015)
        residual = r["residual"][:,0]
        valid = (t>=60)&(t<240)
        events = detect_events(residual,(residual>.01).astype(float),valid,fs,RUN_CONFIG,60)
        assert len(events)==3
        shifted = (residual+3)-3
        shifted_events = detect_events(shifted,(shifted>.01).astype(float),valid,fs,RUN_CONFIG,60)
        np.testing.assert_allclose([e["positive_auc"] for e in events],[e["positive_auc"] for e in shifted_events],atol=1e-14)
        results.append([e["amplitude"] for e in events])
    np.testing.assert_allclose(results,.02,atol=.00015)


def test_gap_invalidates_window_without_bridging():
    q = np.ones((100,2)); q[50,0]=np.nan
    result = correct_signal(q,11)
    assert not result["valid"][45:56,0].any()
    assert result["valid"][:45,0].all()
    assert result["window_support"][0]==6
    assert result["baseline_count"][0]==89
    with pytest.raises(ValueError,match="No valid baseline"):
        correct_signal(np.full((10,1),np.nan),11)


def test_multiplicative_loss_and_true_decline_remain():
    t = np.arange(3000)/10
    pulse = np.zeros(3000)
    pulse[700:710]=.04; pulse[2100:2110]=.04
    bleach = 1-.001*t
    r = correct_signal(((1+pulse)*bleach)[:,None],601)["residual"][:,0]
    assert r[2105]<.9*r[705]  # additive correction must not claim amplitude restoration
    pulse[2100:2110]=.02
    r = correct_signal((1-.0001*t+pulse)[:,None],601)["residual"][:,0]
    assert .45<r[2105]/r[705]<.55


def test_sustained_event_absorbed_and_emo_only_change():
    q = np.ones((3000,1)); q[800:1800]+=.1
    result = correct_signal(q,601)
    assert abs(result["residual"][1300,0])<1e-10
    emo = np.ones((3000,1)); emo[1400:1410]*=.9
    result = correct_signal(np.ones_like(emo)/emo,601)
    assert result["residual"][1405,0]>.1  # reflectance-only glitches can mimic events


def test_hysteresis_merge_gaps_and_exposure():
    trace = np.zeros(100); trace[20:30]=.02; trace[32:40]=.02
    fraction = (trace>0).astype(float)
    valid = np.ones(100,bool)
    events = detect_events(trace,fraction,valid,10,RUN_CONFIG,60)
    assert len(events)==1 and events[0]["duration_s"]==2
    valid[30:32]=False
    events = detect_events(trace,fraction,valid,10,RUN_CONFIG,60)
    assert len(events)==2
    row = metric_row(trace,np.maximum(trace,0),fraction,valid,events,0,100,10)
    assert row["valid_seconds"]==9.8
    assert row["occupied_seconds"]==1.8
    first = metric_row(trace,trace,fraction,valid,events,0,25,10)
    second = metric_row(trace,trace,fraction,valid,events,25,100,10)
    assert first["event_count"]+second["event_count"]==2
    assert first["occupied_seconds"]+second["occupied_seconds"]==row["occupied_seconds"]


def test_noise_false_events_and_spatial_recruitment():
    rng = np.random.default_rng(5)
    q = 1+rng.normal(0,.001,(1500,20))
    q[700:710,:10]+=.03
    res = correct_signal(q,201)["residual"]
    frac = (res>.01).mean(axis=1)
    np.testing.assert_allclose(frac[700:710],.5)
    events = detect_events(res.mean(axis=1),frac,np.ones(1500,bool),10,RUN_CONFIG,20)
    assert len(events)==1


def test_partial_bin_aggregation_weights_exposure():
    df = pd.DataFrame(dict(animal=["a","a"],valid_seconds=[30,10],occupied_seconds=[3,5],event_count=[1,2]))
    result = aggregate_exposure(df,["animal"]).iloc[0]
    assert result.burden==.2
    assert result.events_per_minute==4.5


def test_condition_models_on_clustered_synthetic_data(tmp_path):
    from compare_cortex_activity_conditions import fit_models
    rng = np.random.default_rng(42)
    rows = []
    for gi, group in enumerate(('PV', 'R', 'T')):
        for animal in range(3):
            for day in (1, 2):
                for recording in (1, 2):
                    for time in (75, 105, 135, 165):
                        count = rng.poisson(2+gi)
                        rows.append(dict(group=group, animal=f'{group}{animal}', mouse_line='PV-CRE',
                            day_index=day, recording_index=recording, midpoint_s=time,
                            valid_seconds=30., event_count=count, burden=(count+1)/30))
    contrasts, report = fit_models(pd.DataFrame(rows), 3, rng, tmp_path)
    assert len(contrasts)==36
    assert all(r['status']=='ok' for r in report)
    assert all(np.isfinite(r['estimate']) for r in contrasts)
    assert all(r['status']=='unstable bootstrap' for r in contrasts)


def test_raw_recording_geometry_outputs_and_cache(tmp_path):
    from pathlib import Path
    import json
    import yaml
    from analyze_cortex_activity_emo import process_recording
    folder = tmp_path/'pixels'/'260101_R1'/'t1'
    folder.mkdir(parents=True)
    atlas_path = tmp_path/'roi.yaml'
    atlas = dict(grid=[4,4],bregma_row=2,bregma_col=2,downsample=1,
        boxes=dict(M1L=dict(row_start=0,row_end=0,col_start=-1,col_end=-1),
                   M2L=dict(row_start=0,row_end=0,col_start=-1,col_end=-1),
                   M1R=dict(row_start=0,row_end=0,col_start=1,col_end=1)))
    atlas_path.write_text(yaml.safe_dump(atlas))
    meta = dict(atlas,roi_set=str(atlas_path),axis_order='time,y,x',n_time=300,n_written=300,
        shape=[300,4,4],region=[0,4,0,4],y_1=2,x_2=2)
    meta.pop('boxes')
    np.savez(folder/'pixels_meta_full.npz',**meta)
    signal = np.full((300,4,4),1000,dtype=np.float32)
    np.save(folder/'pixels_f_gcamp_full.npy',signal)
    np.save(folder/'pixels_f_emo_full.npy',signal)
    row = dict(recording_id='260101_R1_t1',pixel_dir='260101_R1/t1',group='R',animal='R1')
    config = dict(RUN_CONFIG,pixel_root=str(tmp_path/'pixels'),output_root=str(tmp_path/'out'),
                  baseline_windows_s=[2.0],analysis_margin_s=2.,skip_start_s=0.,threshold_factors=[1.0],step_fraction=0)
    first = process_recording(row,config)
    assert first['events.csv'].empty
    cortex = first['recording_metrics.csv'].query("region == 'cortex' and support == 'common'").iloc[0]
    assert cortex.mask_pixels==2  # two overlapping left ROIs count once
    assert cortex.valid_seconds==26
    output = tmp_path/'out'/'260101_R1'/'t1'
    with np.load(output/'traces_2s.npz') as traces:
        assert 'equal_roi_residual' in traces
        np.testing.assert_allclose(traces['corrected']-traces['mean_b'],traces['residual'])
    key1 = json.loads((output/'complete.json').read_text())['key']
    timestamp = (output/'traces_2s.npz').stat().st_mtime_ns
    process_recording(row,config)
    assert (output/'traces_2s.npz').stat().st_mtime_ns==timestamp
    process_recording(row,dict(config,pixel_threshold=.02))
    assert json.loads((output/'complete.json').read_text())['key']!=key1
    signal[150,1,0] = 0  # one bad denominator, retained pixel, unequal window dilation
    np.save(folder/'pixels_f_emo_full.npy',signal)
    tables = process_recording(row,dict(config,baseline_windows_s=[2.0,4.0]))
    common = tables['recording_metrics.csv'].query("region == 'cortex' and support == 'common'")
    np.testing.assert_allclose(common.valid_seconds,21.9)
    with np.load(output/'traces_2s.npz') as traces:
        assert traces['valid'][130,0]
        assert not traces['common_valid'][130,0]

    # Windows spawn workers must reproduce the serial tables and isolate output writes.
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from analyze_cortex_activity_emo import process_job, cache_matches
    configs = [dict(config,baseline_windows_s=[2.0,4.0],output_root=str(tmp_path/f'parallel{i}')) for i in range(2)]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        results = [pool.submit(process_job,row,c) for c in configs]
        assert all(f.result()['status']=='computed' for f in results)
    for c in configs:
        actual = pd.read_csv(Path(c['output_root'])/row['pixel_dir']/'recording_metrics.csv')
        pd.testing.assert_frame_equal(actual,tables['recording_metrics.csv'],check_dtype=False)
        assert cache_matches(row,dict(c,workers=4))
    import analyze_cortex_activity_emo as module
    from unittest.mock import patch
    with patch.object(module,'load_recording',side_effect=AssertionError('cache opened raw data')):
        module.process_recording(row,dict(config,baseline_windows_s=[2.0,4.0]))




def test_no_shared_line_skips_models_without_crashing(tmp_path):
    from compare_cortex_activity_conditions import fit_models
    data = pd.DataFrame(dict(group=['PV','R'],animal=['PV1','R1'],mouse_line=['PV-CRE','C57'],
        valid_seconds=[30.,30.],day_index=[1,1],recording_index=[1,1]))
    contrasts,report = fit_models(data,100,np.random.default_rng(1),tmp_path)
    assert contrasts==[]
    assert 'insufficient' in report[0]['status']
