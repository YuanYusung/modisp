#!/Users/yusong/miniforge3/bin/python
"""Benchmark checkpoint integrity and validation/comparison utilities."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import modisp as md
BACKEND=os.environ.get("MODISP_TEST_BACKEND","cpu")
ATTRS=("phase_velocity","mode_mask","status","trace_flags","search_level","root_count","search_bounds")


def test_streaming_complete_resume_and_corruption(tmp_path):
    # Independent NPY output, complete diagnostics and resume integrity.
    tool = Path(__file__).resolve().parents[1] / "tools" / "benchmark_modephys.py"
    if not tool.exists():
        pytest.skip("source-only benchmark driver is not installed in runtime wheel")
    spec = importlib.util.spec_from_file_location("modephys_benchmark", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    model = np.zeros((3, 70, 4), dtype=np.float32)
    model[:, :, 0] = np.arange(70)*.04
    model[:, :, 1:] = [2., 1., 2.]
    np.save(tmp_path / "models.npy", model)
    np.save(tmp_path / "periods.npy", np.array([.4, 1.]))
    args = SimpleNamespace(input=tmp_path / "models.npy", periods=tmp_path / "periods.npy",
                           output=tmp_path / "out", backend=BACKEND, threads=2,
                           workspace_bytes=16*1024**2, native_chunk=None, models=0,
                           profile=True, chunk=2, resume=False)
    driver.run(args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["complete"] and len(report["blocks"]) == 2
    assert report["stats"]["engine_allocations"] == 1
    assert report["unresolved_traces"] == 0
    assert np.load(args.output / "phase_velocity.npy").dtype == np.float64
    assert np.load(args.output / "periods.npy").dtype == np.float64
    assert all((args.output / f"{name}.npy").exists() for name in ATTRS)
    args.resume = True
    args.secular = "fast"
    with pytest.raises(ValueError, match="configuration"):
        driver.run(args)
    args.secular = None
    driver.run(args)
    data = np.load(args.output / "phase_velocity.npy", mmap_mode="r+")
    data[0, 0, 0] += .1
    data.flush()
    with pytest.raises(ValueError, match="checksum"):
        driver.run(args)


def test_interrupted_streaming_resumes_from_verified_block(tmp_path, monkeypatch):
    tool = Path(__file__).resolve().parents[1] / "tools" / "benchmark_modephys.py"
    if not tool.exists():
        pytest.skip("source-only benchmark driver is not installed in runtime wheel")
    spec = importlib.util.spec_from_file_location("modephys_resume", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    model = np.zeros((3, 70, 4), dtype=np.float32)
    model[:, :, 0] = np.arange(70)*.04
    model[:, :, 1:] = [2., 1., 2.]
    np.save(tmp_path / "model.npy", model)
    np.save(tmp_path / "periods.npy", [.5, 1.])
    args = SimpleNamespace(input=tmp_path / "model.npy", periods=tmp_path / "periods.npy",
                           output=tmp_path / "out", backend=BACKEND, threads=2,
                           workspace_bytes=16*1024**2, native_chunk=None, models=0,
                           profile=False, chunk=2, resume=False)
    original = driver.make_solver
    class Interrupted:
        def __init__(self):
            self.solver = original(md, args)
            self.calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            self.solver.close()
        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt("simulated interruption after checkpoint")
            return self.solver(**kwargs)
        def stats(self):
            return self.solver.stats()
    monkeypatch.setattr(driver, "make_solver", lambda *unused: Interrupted())
    with pytest.raises(KeyboardInterrupt):
        driver.run(args)
    state = json.loads((args.output / "checkpoint.json").read_text())
    assert not state["complete"] and state["blocks"][0]["stop"] == 2
    monkeypatch.setattr(driver, "make_solver", original)
    args.resume = True
    driver.run(args)
    state = json.loads((args.output / "report.json").read_text())
    assert state["complete"] and state["resumed"]
    assert "end_to_end_seconds" not in state
    with md.Solver(backend=BACKEND) as solver:
        expected = solver(**driver.convert(model), periods=[.5, 1.], nmodes=5)
    for name in ATTRS:
        np.testing.assert_array_equal(np.load(args.output / f"{name}.npy"), getattr(expected, name))


def test_reference_union_does_not_merge_two_close_modes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"tools"))
    from validate import merge_witnesses
    root=np.array([1.,1.0000001,1.1])
    merged=merge_witnesses(root,[root+1e-10,root-1e-10])
    np.testing.assert_array_equal(merged,root)
    assert len(merged)==3


@pytest.mark.parametrize("background_names", [(), ("ffmpeg",)])
def test_gpu_monitor_records_opted_in_background_but_rejects_other_compute(background_names):
    tool = Path(__file__).resolve().parents[1]/"tools"/"benchmark_modephys.py"
    spec = importlib.util.spec_from_file_location("gpu_monitor_benchmark", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monitor = object.__new__(driver.GpuMonitor)
    monitor.allowed = {100}
    monitor.background_names = set(background_names)
    monitor.background_samples = []
    monitor.competitors = [];monitor.samples = [];monitor.peak_mib = {}
    # A reconnect changes the video PID; neither background process may hide
    # unexpected compute from a different Python job.
    lines = ["0 100 C 40 10 - - - - 700 0 python",
             "0 200 C 2 1 - - - - 300 0 ffmpeg",
             "0 201 C 1 1 - - - - 300 0 ffmpeg",
             "0 300 C 20 10 - - - - 500 0 python"]
    monitor.proc = SimpleNamespace(stdout=lines, poll=lambda: None)
    monitor._read()
    assert len(monitor.samples) == 4 and monitor.peak_mib[100] == 700
    assert len(monitor.background_samples) == (2 if background_names else 0)
    assert len(monitor.competitors) == (1 if background_names else 3)
    with pytest.raises(RuntimeError, match="concurrent GPU compute"):
        monitor.check()


@pytest.mark.parametrize("version", ["0.4.0", "99.0.0"])
def test_baseline_monitor_registers_worker_before_setup_and_cleans_up(version, tmp_path, monkeypatch):
    import io
    tool = Path(__file__).resolve().parents[1]/"tools"/"benchmark_continuation.py"
    spec = importlib.util.spec_from_file_location("continuation_monitor_benchmark", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monitor = SimpleNamespace(allowed={100})
    setup_checks = []
    pid = 431

    class Input(io.StringIO):
        def write(self, value):
            command = json.loads(value)
            # Worker setup may initialize CUDA before its response reaches
            # the parent. Its PID must already be exempt from peer detection.
            assert pid in monitor.allowed
            if command["action"] == "setup":
                setup_checks.append(True)
            return super().write(value)

    class Process:
        def __init__(self):
            self.pid = pid
            self.returncode = None
            self.stdin = Input()
            self.stdout = io.StringIO(json.dumps(dict(version=version))+"\n")
            self.stderr = io.StringIO()

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert pid in monitor.allowed
            self.returncode = 0
            return 0

    process = Process()

    def popen(*args, **kwargs):
        assert pid not in monitor.allowed
        return process

    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    arguments = (tmp_path/"python040", tmp_path/"input.npz", {"backend": "cuda"}, "rayleigh", 1)
    if version == "0.4.0":
        baseline = driver.Baseline(*arguments, monitor=monitor)
        assert baseline.metadata["version"] == version
        assert pid in monitor.allowed
        baseline.close()
    else:
        # Failed setup/version checks must not leave a stale allowed PID.
        with pytest.raises(ValueError, match="0.4.0"):
            driver.Baseline(*arguments, monitor=monitor)
    assert setup_checks == [True]
    assert monitor.allowed == {100}
    assert process.returncode == 0
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))


@pytest.fixture
def phase_verification_case():
    tool = Path(__file__).resolve().parents[1]/"tools"/"benchmark_continuation.py"
    spec = importlib.util.spec_from_file_location("continuation_verification_benchmark", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    B, M, T = 3, 2, 3
    phase = (.5+.1*np.arange(B)[:, None, None]
             +.4*np.arange(M)[None, :, None]+.05*np.arange(T)[None, None, :])

    def snapshot(values):
        return dict(phase_velocity=values.copy(), root_count=np.full((B, T), 4, dtype=np.int32),
                    trace_flags=np.zeros((B, T), dtype=np.uint32))

    case = SimpleNamespace(driver=driver, periods=np.array([1., 1.1, 1.2]),
                           high=snapshot(phase), independent=snapshot(phase),
                           candidate=snapshot(phase), calls=[])
    # A duplicate alias in the old search is a wrong second root. Correcting
    # it is a strict compatibility difference that still needs physical proof.
    case.independent["phase_velocity"][1, 1, 1] = phase[1, 0, 1]*(1+1e-7)
    case.batch = dict(thickness=np.column_stack((np.arange(B)+1., np.zeros(B))),
                      vp=np.full((B, 2), 4.), vs=np.full((B, 2), 2.), rho=np.full((B, 2), 2.))
    case.options = dict(backend="cuda", precision="float32", accuracy="balanced",
                        profile=False, secular="fast", workspace_bytes=123456)

    class HighSolver:
        def __init__(self, **options):
            case.calls.append(dict(options=options))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def __call__(self, **kwargs):
            models = np.rint(kwargs["thickness"][:, 0]-1).astype(int)
            case.calls[-1]["model_indices"] = models.tolist()
            np.testing.assert_array_equal(kwargs["periods"], case.periods)
            assert kwargs["nmodes"] == M
            assert kwargs["wave"] == "rayleigh"
            arrays = {name: value[models].copy() for name, value in case.high.items()}
            return SimpleNamespace(**arrays)

    case.md = SimpleNamespace(Solver=HighSolver, __version__="test")
    case.verify = lambda snapshots: driver.verify_phase_differences(
        case.md, case.batch, case.periods, M, "rayleigh", case.options, snapshots)
    return case


def test_physical_verification_requires_high_proof_without_erasing_strict_difference(phase_verification_case):
    c = phase_verification_case
    strict_before = c.driver.compare_phase(c.independent, c.candidate)
    assert not strict_before["passed"] and strict_before["wrong"] == 1
    result = c.verify(dict(independent=c.independent, continuation=c.candidate))["continuation"]
    assert result["passed"]
    assert result["changed_traces"] == 1
    assert result["model_indices"] == [1]
    assert result["candidate_comparison"]["passed"]
    assert not result["independent_comparison"]["passed"]
    assert c.driver.compare_phase(c.independent, c.candidate) == strict_before
    assert len(c.calls) == 1 and c.calls[0]["model_indices"] == [1]
    assert c.calls[0]["options"]["accuracy"] == "high"
    assert c.calls[0]["options"]["precision"] == "float64"
    assert c.calls[0]["options"].get("frequency_strategy", "independent") == "independent"
    assert c.options["precision"] == "float32" and c.options["accuracy"] == "balanced"


@pytest.mark.parametrize("failure", ["candidate_wrong", "both_wrong", "missing", "extra", "count", "infinite_missing"])
def test_physical_verification_rejects_wrong_or_incomplete_candidate_prefix(phase_verification_case, failure):
    c = phase_verification_case
    if failure == "candidate_wrong":
        c.candidate["phase_velocity"][1, 1, 1] += .2
    elif failure == "both_wrong":
        # The first mode is wrong in both variants and therefore is absent
        # from the strict changed-cell mask. Verify the whole changed trace.
        c.independent["phase_velocity"][1, 0, 1] += .1
        c.candidate["phase_velocity"][1, 0, 1] += .1
    elif failure == "missing":
        c.candidate["phase_velocity"][1, 1, 1] = np.nan
    elif failure == "extra":
        c.high["phase_velocity"][1, 1, 1] = np.nan
        c.high["root_count"][1, 1] = 1
    elif failure == "infinite_missing":
        c.high["phase_velocity"][1, 1, 1] = np.nan
        c.candidate["phase_velocity"][1, 1, 1] = np.inf
        c.high["root_count"][1, 1] = c.candidate["root_count"][1, 1] = 1
    else:
        c.candidate["root_count"][1, 1] = 3
    result = c.verify(dict(independent=c.independent, continuation=c.candidate))["continuation"]
    assert not result["passed"], result


@pytest.mark.parametrize("failure", ["negative_count", "unresolved_flag", "nonascending", "prefix_hole", "infinite_missing"])
def test_physical_verification_rejects_unresolved_or_invalid_high_reference(phase_verification_case, failure):
    c = phase_verification_case
    if failure == "negative_count":
        c.high["root_count"][1, 1] = -1
    elif failure == "unresolved_flag":
        c.high["trace_flags"][1, 1] = md.TRACE_SEARCH_UNCONVERGED
    elif failure == "nonascending":
        c.high["phase_velocity"][1, :, 1] = c.high["phase_velocity"][1, ::-1, 1]
        c.candidate["phase_velocity"][1, :, 1] = c.high["phase_velocity"][1, :, 1]
    elif failure == "infinite_missing":
        c.high["phase_velocity"][1, 1, 1] = np.inf
        c.candidate["phase_velocity"][1, 1, 1] = np.nan
        c.high["root_count"][1, 1] = c.candidate["root_count"][1, 1] = 1
    else:
        c.high["phase_velocity"][1, 0, 1] = np.nan
        c.candidate["phase_velocity"][1, 0, 1] = np.nan
        c.high["root_count"][1, 1] = c.candidate["root_count"][1, 1] = 1
    result = c.verify(dict(independent=c.independent, continuation=c.candidate))["continuation"]
    assert not result["passed"], result


def test_physical_verification_ignores_unchanged_traces_and_batches_shared_references(phase_verification_case):
    c = phase_verification_case
    # A reference failure at an unchanged frequency must not expand the scope
    # into validating legacy outputs that the new feature did not alter.
    c.high["phase_velocity"][1, :, 0] = np.nan
    c.high["root_count"][1, 0] = -1
    c.high["trace_flags"][1, 0] = md.TRACE_SEARCH_UNCONVERGED
    c.independent["phase_velocity"][2, 1, 2] = .8
    first = {k: v.copy() for k, v in c.independent.items()}
    first["phase_velocity"][1, :, 1] = c.candidate["phase_velocity"][1, :, 1]
    second = {k: v.copy() for k, v in c.independent.items()}
    second["phase_velocity"][2, :, 2] = c.candidate["phase_velocity"][2, :, 2]
    result = c.verify(dict(independent=c.independent, continuation=first, combined=second))
    assert result["continuation"]["passed"] and result["combined"]["passed"]
    assert result["continuation"]["changed_traces"] == result["combined"]["changed_traces"] == 1
    assert len(c.calls) == 1
    assert c.calls[0]["model_indices"] == [1, 2]


def test_physical_verification_does_no_work_when_strict_outputs_match(phase_verification_case):
    c = phase_verification_case
    same = {k: v.copy() for k, v in c.independent.items()}
    result = c.verify(dict(independent=c.independent, group=same))
    assert result == {}
    assert not c.calls


def test_benchmark_rejects_crossing_and_finite_mismatches():
    import importlib.util
    tool = Path(__file__).resolve().parents[1] / "tools" / "benchmark_secular.py"
    if not tool.exists():
        pytest.skip("source-only benchmark driver is not installed in runtime wheel")
    spec = importlib.util.spec_from_file_location("secular_benchmark", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    c = np.array([1., 2., 3.])
    f = np.array([-1., 1., 1.])
    good = driver.compare_pointwise(c, f, f.copy())
    assert good["accepted"] and good["max_crossing_node_difference"] == 0.
    missing = driver.compare_pointwise(c, f, np.ones(3))
    assert not missing["accepted"] and not missing["crossing_count_equal"]
    assert missing["max_crossing_node_difference"] is None
    invalid = driver.compare_pointwise(c, f, np.array([-1., np.nan, 1.]))
    assert not invalid["accepted"] and not invalid["finite_mask_equal"]
    invalid = driver.compare_pointwise(c, np.full(3, np.nan), np.full(3, np.nan))
    assert not invalid["accepted"] and invalid["max_abs_difference"] is None


def test_consolidated_validation_stress_entrypoint(tmp_path):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
    import validate
    output=tmp_path/"validation"
    code=validate.main(["stress","--output",str(output),"--backend",BACKEND,
                        "--seeds","3490","--models","1","--layers","4","--periods","3"])
    report=json.loads((output/"report.json").read_text())
    assert code==0 and report["passed"]
    assert {row["wave"] for row in report["cases"]}=={"rayleigh","love"}
    assert all(row["secular"]==("fast" if BACKEND=="cuda" else "dunkin") for row in report["cases"])
