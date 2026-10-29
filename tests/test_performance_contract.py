#!/Users/yusong/miniforge3/bin/python
"""Numerical and lifecycle regressions introduced by workspace/stream reuse."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import modisp as md

BACKEND = os.environ.get("MODISP_TEST_BACKEND", "cpu")
FIELDS = ("thickness", "vp", "vs", "rho")
ATTRS = ("phase_velocity", "mode_mask", "status", "trace_flags", "search_level",
         "root_count", "search_bounds")


def same(a, b, selection=slice(None)):
    for name in ATTRS:
        np.testing.assert_array_equal(getattr(a, name)[selection], getattr(b, name))


def test_reused_capacity_mutated_inputs_and_profile():
    batch = md.random_model_batch(9, 6, seed=20261028)
    periods = np.array([2., .8, 2., 1.4])
    with md.Solver(backend=BACKEND, threads=4) as solver:
        first = solver(**batch, periods=periods, nmodes=3)
        assert solver.stats()["engine_allocations"] == 1
        assert "timings_s" not in solver.stats()
        tail = solver(**{k: v[:2] for k, v in batch.items()}, periods=periods, nmodes=3)
        assert solver.stats()["engine_allocations"] == 0
        assert solver.stats()["capacity_models"] == 9
        same(first, tail, slice(0, 2))
        # Both model and period objects keep their addresses but change values.
        for key in ("vp", "vs"):
            batch[key] *= 1.03
        periods[:] = [1., 3., 1., .5]
        solver.profile = True
        changed = solver(**batch, periods=periods, nmodes=3)
        stats = solver.stats()
        assert stats["timings_s"]["solver_total"] > 0
        assert stats["cache_hits"] >= 0
        stats["timings_s"]["solver_total"] = -1
        assert solver.stats()["timings_s"]["solver_total"] > 0
        with md.Solver(backend=BACKEND, threads=1) as fresh:
            expected = fresh(**batch, periods=periods, nmodes=3)
        same(changed, expected)
        assert np.any(first.phase_velocity != changed.phase_velocity)
        solver.close()
        again = solver(**batch, periods=periods, nmodes=3)
        same(again, expected)


def test_concurrent_solvers_and_result_ownership():
    batch = md.random_model_batch(3, 5, seed=876)
    def work(profile):
        with md.Solver(backend=BACKEND, threads=2, profile=profile) as s:
            return s(**batch, periods=[.5, 1., 2.], nmodes=3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(work, [True, False]))
    same(a, b)
    # A subsequent call/close must not mutate already returned arrays.
    snapshot = a.phase_velocity.copy()
    work(False)
    np.testing.assert_array_equal(snapshot, a.phase_velocity)


@pytest.mark.skipif(BACKEND != "cuda", reason="CUDA launch variants require CUDA")
@pytest.mark.parametrize("block", ["32", "64", "128", "256"])
def test_cuda_launch_profile_and_tail(monkeypatch, block):
    batch = md.random_model_batch(5, 7, seed=545)
    with md.Solver(backend="cuda") as s:
        expected = s(**batch, periods=[.7, 2., .7], nmodes=3)
    monkeypatch.setenv("MODISP_CUDA_BLOCK", block)
    with md.Solver(backend="cuda") as s:
        for profile in (False, False, True, False):
            s.profile = profile
            same(expected, s(**batch, periods=[.7, 2., .7], nmodes=3))
        tail = s(**{k: v[:2] for k, v in batch.items()}, periods=[.7, 2., .7], nmodes=3)
        same(expected, tail, slice(0, 2))
        assert s.stats()["engine_allocations"] == 0


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


def test_maximum_layer_count_shared_memory():
    h=np.zeros(256);vp=np.full(256,6.);vs=np.full(256,3.5);rho=np.full(256,2.7)
    with md.Solver(backend=BACKEND) as s:
        many=s(h,vp,vs,rho,[1.],nmodes=1)
        one=s([0.],[6.],[3.5],[2.7],[1.],nmodes=1)
    assert not np.any(many.trace_flags & md.TRACE_SEARCH_UNCONVERGED)
    np.testing.assert_allclose(many.phase_velocity,one.phase_velocity,atol=3e-6,rtol=2e-6)


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
