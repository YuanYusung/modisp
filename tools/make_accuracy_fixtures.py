#!/Users/yusong/miniforge3/bin/python
"""Capture a compact bitwise baseline from an installed modisp 0.2.1 build.

Run this *before* changing the solver when preparing the 0.3.0 accuracy-mode
regression fixtures.  The script imports the installed ``modisp`` package (not
the source tree) and writes one ``.npz`` per case plus ``manifest.json``.

Examples
--------
CPU (macOS/Linux):
    /path/to/venv-0.2.1/bin/python tools/make_accuracy_fixtures.py \
        --backend cpu --output tests/data/accuracy_high_cpu

CUDA (A800):
    CUDA_VISIBLE_DEVICES=1 /path/to/venv-0.2.1/bin/python \
        tools/make_accuracy_fixtures.py --backend cuda \
        --output tests/data/accuracy_high_cuda
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np


FIELDS = ("phase_velocity", "mode_mask", "status", "trace_flags",
          "search_level", "root_count", "search_bounds")
MODEL_FIELDS = ("thickness", "vp", "vs", "rho")


def _model(name, h, vp, vs, rho):
    return dict(name=name, thickness=np.asarray(h, float), vp=np.asarray(vp, float),
                vs=np.asarray(vs, float), rho=np.asarray(rho, float))


def _canonical(md, name):
    for model in md.CANONICAL_MODELS:
        if model.name == name:
            return _model(name, model.thickness, model.vp, model.vs, model.rho)
    raise KeyError(name)


def cases(md):
    """Deterministic, synthetic cases covering P-SV, water, Love and 70 layers."""
    out = []
    random3 = md.random_model_batch(3, 8, seed=20261028)
    out.append(dict(name="random8", **{k: random3[k] for k in MODEL_FIELDS},
                    periods=[0.2, 0.4, 0.63, 1.0, 2.0, 10.0], nmodes=5,
                    wave="rayleigh"))
    for name in ("m04_lvz", "m06_multi_reversal", "m08_near_kissing"):
        model = _canonical(md, name)
        out.append(dict(name=name, **{k: model[k] for k in MODEL_FIELDS},
                        periods=[0.2, 0.63, 1.0, 2.0, 10.0], nmodes=4,
                        wave="rayleigh"))
    water = _model(
        "water",
        [0.025, 0.0],
        [1.5, 6.0],
        [0.0, 3.5],
        [1.02, 2.7],
    )
    out.append(dict(name="water", **{k: water[k] for k in MODEL_FIELDS},
                    periods=[0.2, 0.63, 1.0, 2.0, 10.0], nmodes=4,
                    wave="rayleigh"))
    solid = _canonical(md, "m02_two_layer")
    out.append(dict(name="love_two_layer", **{k: solid[k] for k in MODEL_FIELDS},
                    periods=[0.2, 0.4, 0.63, 1.0, 2.0, 10.0], nmodes=4,
                    wave="love"))
    deep = md.random_model_batch(2, 70, seed=20261029)
    out.append(dict(name="random70", **{k: deep[k] for k in MODEL_FIELDS},
                    periods=[0.2, 0.4, 0.63, 1.0, 2.0, 10.0], nmodes=5,
                    wave="rayleigh", cmin=0.2, cmax=2.5))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    md = importlib.import_module("modisp")
    if getattr(md, "__version__", None) != "0.2.1":
        raise SystemExit(
            f"expected an installed modisp 0.2.1 baseline, found {md.__version__!r}; "
            "run this with the 0.2.1 release environment")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        modisp_version=md.__version__,
        backend=args.backend,
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        cpu_module=os.path.basename(md._cpu.__file__),
        cpu_sha256=sha256(md._cpu.__file__),
        cuda_module=None,
        cuda_device=None,
        cases=[],
    )
    if args.backend == "cuda":
        try:
            manifest["cuda_device"] = md.device_name()
            program = importlib.import_module("modisp_cuda").load()
            manifest["cuda_module"] = os.path.basename(program.__file__)
            manifest["cuda_sha256"] = sha256(program.__file__)
        except Exception as exc:  # noqa: BLE001 - provenance is diagnostic
            manifest["cuda_probe_error"] = repr(exc)

    for index, case in enumerate(cases(md)):
        options = dict(backend=args.backend, threads=args.threads)
        if args.backend == "cuda":
            options["workspace_bytes"] = 1024 ** 3
        if "cmin" in case:
            options["cmin"] = case["cmin"]
        if "cmax" in case:
            options["cmax"] = case["cmax"]
        with md.Solver(**options) as solver:
            result = solver(**{k: case[k] for k in MODEL_FIELDS},
                            periods=np.asarray(case["periods"], float),
                            nmodes=int(case["nmodes"]), wave=case["wave"])
        arrays = {name: getattr(result, name) for name in FIELDS}
        path = args.output / f"{index:02d}_{case['name']}.npz"
        np.savez_compressed(
            path,
            thickness=np.asarray(case["thickness"], float),
            vp=np.asarray(case["vp"], float),
            vs=np.asarray(case["vs"], float),
            rho=np.asarray(case["rho"], float),
            periods=np.asarray(case["periods"], float),
            nmodes=np.asarray(case["nmodes"], np.int64),
            wave=np.asarray(case["wave"]),
            cmin=np.asarray(case.get("cmin", np.nan), float),
            cmax=np.asarray(case.get("cmax", np.nan), float),
            **arrays,
        )
        manifest["cases"].append(dict(
            file=path.name,
            name=case["name"],
            nmodes=int(case["nmodes"]),
            wave=case["wave"],
            periods=list(map(float, case["periods"])),
            cmin=case.get("cmin"),
            cmax=case.get("cmax"),
            sha256=sha256(path),
        ))
        print(f"wrote {path}", flush=True)

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
