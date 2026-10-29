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


def test_reference_union_does_not_merge_two_close_modes():
    from validate import merge_witnesses
    root=np.array([1.,1.0000001,1.1])
    merged=merge_witnesses(root,[root+1e-10,root-1e-10])
    np.testing.assert_array_equal(merged,root)
    assert len(merged)==3


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
