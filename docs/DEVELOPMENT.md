# Development workflows

## Backend defaults

`Solver(secular=None)` selects the evaluator after resolving the backend:

| Actual backend / path | Default evaluator |
|---|---|
| CUDA adaptive (shape or QE) | `fast` |
| CPU adaptive | `dunkin` |
| Legacy CUDA uniform | `dunkin` |

`backend="auto"` uses fast when CUDA is available and Dunkin on CPU fallback.
An explicit `secular="dunkin"` or `"fast"` overrides adaptive defaults; uniform
rejects explicit fast. Love continues to execute SH. `Solver.secular` stores
the requested option, while `DispersionBatch.secular`, stats, kernel reports
and CLI NPZs record the resolved evaluator. Before the first solve, an auto
solver's kernel report has no resolved evaluator. `modisp doctor` reports both
the backend policy and the default for the currently available auto backend.

## One regression suite

```sh
python -m pytest -q
MODISP_TEST_BACKEND=cuda python -m pytest -q
```

| Module | Retained coverage |
|---|---|
| `tests/test_api.py` | API, CPU/CUDA/auto defaults, fallback, CLI metadata, thread/workspace reuse, concurrency and independent witness inputs |
| `tests/test_physics.py` | Analytic Rayleigh/Love/Scholte checks, disba, precision, scaling and conditioning |
| `tests/test_solver.py` | Accuracy/search presets, frozen Dunkin outputs, paired fast roots, near-kissing pairs, water/LVZ and shared-memory boundaries |
| `tests/test_tools.py` | Checkpoint resume/corruption, comparison failure detection, independent witness union and the unified stress entrypoint |

Seven test modules became four. The seven frozen cases now check both exact
Dunkin outputs and paired fast results in the same execution group. Duplicate
thread/chunk checks and the separate sampling CLI test were folded into the
API/lifecycle checks. Independent physical witnesses and difficult numerical
boundaries remain. Corrected an ordinary-root test that prepared a Love case
but failed to pass `wave="love"` to the solver.

## Extended validation

```sh
python tools/validate.py stress --backend cpu --output /path/to/new-output
python tools/validate.py stress --backend cuda --seeds 3490 3491 --models 16 --periods 12 --output /path/to/new-output
python tools/validate.py paper --backend cuda --grt /path/to/grt --output /path/to/paper-output
python tools/validate.py external --backend cuda --references /path/to/frozen-npz --output /path/to/new-output
```

Each mode has `--help` and accepts an explicit `--secular`. Stress compares
presets with high as a regression witness, retaining unjudged/cluster counts;
it is not an independent physical reference. Paper retains the pinned GRT/raw
witness construction and comparison reports. External reads independent frozen
archives, retaining stability masks and independent-prefix limits. External
archives and third-party executables are not bundled.

## Retained tools

Python tool files were reduced from 19 to 10:

| Tool | Purpose |
|---|---|
| `validate.py` | Paper witnesses, external archives and seeded stress |
| `benchmark_secular.py` | Fixed-node evaluator and paired solver measurements |
| `benchmark_modephys.py` | Large NPY inputs, streaming checkpoints and comparisons |
| `benchmark_ab.py` | Persistent paired before/after installations |
| `benchmark_external.py` | Accuracy/performance against frozen solid-Rayleigh references |
| `external_reference.py` | Build/verify pinned external reference adapters |
| `tolerant_metrics.py` | Shared close-cluster and unjudged-row comparison logic |
| `make_accuracy_fixtures.py` | Recreate fixtures using the explicitly required 0.2.1 baseline |
| `prepare_release.py` | Synchronize the self-contained plugin C++ source tree |
| `audit_release.py` | Source/distribution provenance and packaging audit |

The two C++ helpers remain: `check_secular_algebra.cpp` checks the contraction;
`qed_reference_adapter.cpp` is the external-reference adapter. Generic
benchmarks accept evaluator selection, so new versions do not need another
benchmark wrapper.

Removed version-specific preset/shape benchmarks, duplicate shape/tolerant
stress runners, the one-off sampling scan/adjudication scripts and the separate
installed smoke script. Their reports remain historical records; the original
scripts are recoverable at commit `e6bd811` for exact historical reproduction.
Current work should use the commands above.

## Executed checks for the consolidated tree

- macOS arm64 CPU: 197 passed, 5 CUDA-only skips.
- Linux x86-64 CPU: 197 passed, 5 CUDA-only skips.
- A800/sm_80 CUDA with the new default: 202 passed.
- Unified CUDA stress: seeds 3490/3491, 16 models × 12 periods per wave, passed.
- Generic A/B benchmark: explicit fast + QE selection passed on a small CPU batch.
- Unified external validator: independent analytic half-space archive, all gates passed.

The suite exercises actual backend selection and the same-instance CUDA-to-CPU
fallback, explicit overrides, uniform compatibility and CLI/doctor metadata.
Frozen Dunkin checks explicitly request Dunkin; ordinary CUDA tests exercise
the new fast default. These checks retain the physical and architecture limits
in [the fast-evaluator report](SECULAR_FAST_0_3_2.md).
