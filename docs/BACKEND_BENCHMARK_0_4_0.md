# modisp 0.4.0: A800 and RTX 5050 backend benchmark

Measured on 2026-11-12 (Asia/Shanghai), using the released 0.4.0 CPU/CUDA wheels and source commit `b27734f`; measurement tools are frozen at `6259e53`. Both platforms used the same frozen input files, not merely the same random seed. Each completed 57 path configurations / 19 workload-wave rows.

All four installed wheel hashes match the [published release asset digests](benchmarks/backend-0.4.0-release-wheels.json).

The [prepared-input manifest](benchmarks/backend-0.4.0-inputs.json) records every frozen file hash, including the normalized matrix JSON written by prepare.

**The A800 host and selected GPU had competing workloads. Its times are retained as shared-load diagnostic observations; they are not an uncontended CPU/GPU or cross-platform ranking. RTX 5050 results retain their own monitoring record and an 8-thread CPU baseline.**

## Protocol

The matrix is [machine readable](benchmarks/backend-0.4.0-matrix.json). B=1/64/2048 and T=1/64 form the core 24-layer, five-mode balanced grid for dry Rayleigh and Love. Additional rows isolate K=1, L=70 and high accuracy at B=64/T=64, plus wet Rayleigh/high. L includes the half-space. Modes are ascending recovered roots, not tracked physical branches.

Inputs: 2,048 distinct 24-layer models (seed 20261112), 64 distinct 70-layer models (20261113), h=.04–.25 km, LVZ probability .5, Vs=.6–3.6 km/s and the fully specified generator options in the matrix. Smaller B values are master-batch prefixes. Wet replaces the first layer with (.08,1.5,0,1.02). T=1 is [3] s; T=64 is geometric .15–10 s. Both frozen arrays and hashes are retained.

Settings: shape_adaptive, Brent, Love phase, automatic velocity bounds, close_root_rtol=.005, f_accept=1, 1 GiB workspace budget, no explicit chunk cap. CPU uses Dunkin/FP64; CUDA uses fast with FP64 or FP32. Thus CPU/GPU comparison includes evaluator defaults. Inputs remain contiguous host FP64 for all paths. G32 high includes any FP64 repair attempts.

One first call and three warmups precede 12 calls per path, using all six C64/G64/G32 permutations twice. Values below are complete synchronous Solver-call wall times: automatic bounds, input conversion, transfers, search/repair, native chunks and output packaging. File IO, hashing, continuity checks and profile=True are excluded. The 12 raw samples are in the receipts. Cells show median [Q25,Q75] ms, with linear quantiles. Ratios divide the medians in the same row.

## Hardware and provenance

| Platform | CPU / threads | GPU | Python / NumPy / SciPy | Driver | Timing receipt |
|---|---|---|---|---|---|
| a800 | Intel(R) Xeon(R) Gold 6430 / 32 | NVIDIA A800 80GB PCIe | 3.10.21 / 2.2.6 / 1.15.3 | 560.35.05 | [JSON](benchmarks/backend-0.4.0-20261112-a800.json) |
| rtx5050 | AMD Ryzen 7 9700X 8-Core Processor / 8 | NVIDIA GeForce RTX 5050 | 3.12.3 / 2.2.6 / 1.18.1 | 580.178.04 | [JSON](benchmarks/backend-0.4.0-20261112-rtx5050.json) |

Compact discrepancy snapshots are available for [a800](benchmarks/backend-0.4.0-witnesses-a800.npz), [rtx5050](benchmarks/backend-0.4.0-witnesses-rtx5050.npz). They contain the actual selected model/period inputs, three-path outputs, high-FP64 reference prefixes, status and trace diagnostics. Selection is for inspection; all benchmark counts use complete outputs. Singleton replays may alter FP64 repair quotas, so reproduce the full matrix for the original batch context.

## a800: timing

Foreign GPU activity was observed in 19/19 workload groups. Host one-minute load ranged from 148.79 to 216.45. Rows marked † are diagnostic observations, not accepted uncontended speedups; unmarked rows retain their own monitoring record.

| Case | B × L × T; K / accuracy | CPU FP64 ms | CUDA FP64 ms | CUDA FP32 ms | GPU64 / CPU × | GPU32 / CPU × | FP32 / FP64 × |
|---|---|---:|---:|---:|---:|---:|---:|
| M01-R† | 1 × 24 × 1; 5 / balanced | 3.380 [3.334,3.484] | 5.078 [5.038,5.339] | 4.980 [4.757,5.642] | 0.67 | 0.68 | 1.02 |
| M01-L† | 1 × 24 × 1; 5 / balanced | 1.102 [1.055,1.160] | 4.381 [4.198,6.248] | 2.185 [2.122,2.256] | 0.25 | 0.50 | 2.01 |
| M02-R† | 1 × 24 × 64; 5 / balanced | 22.052 [21.384,26.895] | 5.346 [5.263,5.440] | 5.576 [5.330,6.241] | 4.12 | 3.95 | 0.96 |
| M02-L† | 1 × 24 × 64; 5 / balanced | 10.184 [8.798,11.340] | 9.086 [8.684,9.227] | 2.360 [2.276,2.401] | 1.12 | 4.32 | 3.85 |
| M03-R† | 64 × 24 × 1; 5 / balanced | 12.389 [10.656,13.627] | 5.622 [5.521,5.888] | 9.698 [9.069,10.042] | 2.20 | 1.28 | 0.58 |
| M03-L† | 64 × 24 × 1; 5 / balanced | 9.527 [7.382,11.743] | 7.636 [6.580,7.936] | 2.367 [2.333,2.412] | 1.25 | 4.03 | 3.23 |
| M04-R† | 64 × 24 × 64; 5 / balanced | 360.423 [355.714,364.045] | 29.846 [29.226,32.253] | 40.517 [39.990,47.145] | 12.08 | 8.90 | 0.74 |
| M04-L† | 64 × 24 × 64; 5 / balanced | 135.482 [134.472,137.674] | 26.408 [25.596,27.740] | 12.192 [10.546,12.578] | 5.13 | 11.11 | 2.17 |
| M05-R† | 2048 × 24 × 1; 5 / balanced | 195.779 [192.004,201.211] | 22.847 [21.567,26.799] | 32.559 [32.069,33.627] | 8.57 | 6.01 | 0.70 |
| M05-L† | 2048 × 24 × 1; 5 / balanced | 59.248 [56.149,74.400] | 15.417 [13.950,15.480] | 9.113 [8.354,9.202] | 3.84 | 6.50 | 1.69 |
| M06-R† | 2048 × 24 × 64; 5 / balanced | 9871.471 [9725.178,10121.387] | 890.594 [877.163,907.975] | 1072.039 [1063.429,1094.490] | 11.08 | 9.21 | 0.83 |
| M06-L† | 2048 × 24 × 64; 5 / balanced | 3155.564 [3091.693,3247.217] | 609.703 [587.136,626.924] | 254.362 [252.914,256.814] | 5.18 | 12.41 | 2.40 |
| S01-R† | 64 × 24 × 64; 1 / balanced | 300.324 [288.816,314.959] | 22.130 [20.715,23.097] | 33.711 [27.505,35.582] | 13.57 | 8.91 | 0.66 |
| S01-L† | 64 × 24 × 64; 1 / balanced | 121.566 [106.711,134.704] | 22.776 [21.571,23.611] | 11.462 [11.342,11.646] | 5.34 | 10.61 | 1.99 |
| S02-R† | 64 × 70 × 64; 5 / balanced | 1306.305 [1271.780,1329.465] | 87.116 [81.394,90.795] | 95.485 [84.557,99.309] | 15.00 | 13.68 | 0.91 |
| S02-L† | 64 × 70 × 64; 5 / balanced | 613.981 [607.594,627.709] | 72.624 [71.483,80.815] | 32.182 [31.889,38.242] | 8.45 | 19.08 | 2.26 |
| S03-R† | 64 × 24 × 64; 5 / high | 846.227 [823.257,970.637] | 119.961 [113.682,120.840] | 253.291 [245.106,256.510] | 7.05 | 3.34 | 0.47 |
| S03-L† | 64 × 24 × 64; 5 / high | 188.853 [174.871,198.890] | 40.185 [38.568,42.007] | 32.011 [27.491,34.669] | 4.70 | 5.90 | 1.26 |
| W01-WR† | 64 × 24 × 64; 5 / high | 729.544 [716.312,743.184] | 122.323 [118.854,132.081] | 663.301 [645.753,671.152] | 5.96 | 1.10 | 0.18 |

R=dry Rayleigh, L=Love, WR=wet Rayleigh. M01–M06 are the B/T grid, S01 is K=1, S02 is L=70 and S03 is high. Raw timing ratios do not imply equivalent root completeness. The numerical tables below retain mismatches and failed qualification.

### a800: throughput and sampled memory

Rates use the steady median; traces/s=B*T/t and do not multiply by K. PID memory is measured separately with one CUDA worker, requested 10 ms NVML sampling. It includes process context overhead and is a sampled lower bound on instantaneous peak. Actual sampling intervals and native chunk capacities are in the receipt; native device bytes and workspace budgets are not total VRAM peaks.

| Case / path | models/s | traces/s | VALID roots/call | valid roots/s | PID sampled peak MiB | Native device MiB | Chunks / capacity |
|---|---:|---:|---:|---:|---:|---:|---|
| M01-R / C64 | 295.87 | 295.87 | 3 | 887.62 | N/A | N/A | 1 / 1 |
| M01-R / G64 | 196.92 | 196.92 | 3 | 590.76 | 418 | 0.0375 | 1 / 1 |
| M01-R / G32 | 200.78 | 200.78 | 3 | 602.35 | 420 | 0.03806 | 1 / 1 |
| M01-L / C64 | 907.71 | 907.71 | 2 | 1815.43 | N/A | N/A | 1 / 1 |
| M01-L / G64 | 228.25 | 228.25 | 2 | 456.50 | 420 | 0.0375 | 1 / 1 |
| M01-L / G32 | 457.66 | 457.66 | 2 | 915.32 | 420 | 0.03806 | 1 / 1 |
| M02-R / C64 | 45.35 | 2902.29 | 251 | 11382.44 | N/A | N/A | 1 / 1 |
| M02-R / G64 | 187.06 | 11971.72 | 251 | 46951.58 | 418 | 0.3172 | 1 / 1 |
| M02-R / G32 | 179.34 | 11477.76 | 251 | 45014.32 | 420 | 0.3178 | 1 / 1 |
| M02-L / C64 | 98.19 | 6284.20 | 220 | 21601.95 | N/A | N/A | 1 / 1 |
| M02-L / G64 | 110.06 | 7043.62 | 220 | 24212.45 | 420 | 0.3172 | 1 / 1 |
| M02-L / G32 | 423.71 | 27117.13 | 220 | 93215.13 | 420 | 0.3178 | 1 / 1 |
| M03-R / C64 | 5166.01 | 5166.01 | 162 | 13076.47 | N/A | N/A | 1 / 64 |
| M03-R / G64 | 11384.79 | 11384.79 | 162 | 28817.76 | 420 | 2.38 | 1 / 64 |
| M03-R / G32 | 6599.60 | 6599.60 | 162 | 16705.23 | 422 | 2.416 | 1 / 64 |
| M03-L / C64 | 6717.48 | 6717.48 | 109 | 11440.71 | N/A | N/A | 1 / 64 |
| M03-L / G64 | 8381.38 | 8381.38 | 109 | 14274.55 | 422 | 2.38 | 1 / 64 |
| M03-L / G32 | 27042.74 | 27042.74 | 109 | 46057.16 | 422 | 2.416 | 1 / 64 |
| M04-R / C64 | 177.57 | 11364.44 | 14765 | 40965.81 | N/A | N/A | 1 / 64 |
| M04-R / G64 | 2144.34 | 137237.87 | 14765 | 494706.33 | 438 | 20.25 | 1 / 64 |
| M04-R / G32 | 1579.58 | 101093.38 | 14765 | 364414.99 | 440 | 20.29 | 1 / 64 |
| M04-L / C64 | 472.39 | 30232.76 | 13147 | 97038.59 | N/A | N/A | 1 / 64 |
| M04-L / G64 | 2423.50 | 155104.26 | 13147 | 497840.75 | 440 | 20.25 | 1 / 64 |
| M04-L / G32 | 5249.29 | 335954.68 | 13147 | 1078319.39 | 440 | 20.29 | 1 / 64 |
| M05-R / C64 | 10460.80 | 10460.80 | 5215 | 26637.24 | N/A | N/A | 1 / 2048 |
| M05-R / G64 | 89641.47 | 89641.47 | 5215 | 228261.86 | 494 | 76.16 | 1 / 2048 |
| M05-R / G32 | 62900.55 | 62900.55 | 5215 | 160169.11 | 496 | 77.29 | 1 / 2048 |
| M05-L / C64 | 34566.44 | 34566.44 | 3476 | 58668.44 | N/A | N/A | 1 / 2048 |
| M05-L / G64 | 132841.94 | 132841.94 | 3476 | 225468.06 | 496 | 76.16 | 1 / 2048 |
| M05-L / G32 | 224737.16 | 224737.16 | 3476 | 381438.66 | 496 | 77.29 | 1 / 2048 |
| M06-R / C64 | 207.47 | 13277.86 | 473743 | 47991.13 | N/A | N/A | 1 / 2048 |
| M06-R / G64 | 2299.59 | 147173.64 | 473743 | 531940.31 | 920 | 490.8 | 2 / 1551 |
| M06-R / G32 | 1910.38 | 122264.24 | 473773 | 441936.45 | 922 | 491 | 2 / 1549 |
| M06-L / C64 | 649.01 | 41536.80 | 421616 | 133610.36 | N/A | N/A | 1 / 2048 |
| M06-L / G64 | 3359.01 | 214976.93 | 421616 | 691510.88 | 922 | 490.8 | 2 / 1551 |
| M06-L / G32 | 8051.51 | 515296.45 | 421618 | 1657548.98 | 922 | 491 | 2 / 1549 |
| S01-R / C64 | 213.10 | 13638.59 | 4096 | 13638.59 | N/A | N/A | 1 / 64 |
| S01-R / G64 | 2891.97 | 185085.95 | 4096 | 185085.95 | 436 | 19.13 | 1 / 64 |
| S01-R / G32 | 1898.51 | 121504.64 | 4096 | 121504.64 | 438 | 19.16 | 1 / 64 |
| S01-L / C64 | 526.46 | 33693.53 | 4096 | 33693.53 | N/A | N/A | 1 / 64 |
| S01-L / G64 | 2809.96 | 179837.63 | 4096 | 179837.63 | 438 | 19.13 | 1 / 64 |
| S01-L / G32 | 5583.76 | 357360.61 | 4096 | 357360.61 | 438 | 19.16 | 1 / 64 |
| S02-R / C64 | 48.99 | 3135.56 | 18659 | 14283.80 | N/A | N/A | 1 / 64 |
| S02-R / G64 | 734.66 | 47017.97 | 18659 | 214186.58 | 438 | 20.43 | 1 / 64 |
| S02-R / G32 | 670.27 | 42896.99 | 18659 | 195413.79 | 440 | 20.54 | 1 / 64 |
| S02-L / C64 | 104.24 | 6671.22 | 17274 | 28134.42 | N/A | N/A | 1 / 64 |
| S02-L / G64 | 881.25 | 56399.79 | 17274 | 237853.99 | 440 | 20.43 | 1 / 64 |
| S02-L / G32 | 1988.67 | 127274.72 | 17274 | 536753.78 | 440 | 20.54 | 1 / 64 |
| S03-R / C64 | 75.63 | 4840.31 | 14765 | 17448.03 | N/A | N/A | 1 / 64 |
| S03-R / G64 | 533.51 | 34144.56 | 14765 | 123082.15 | 438 | 20.25 | 1 / 64 |
| S03-R / G32 | 252.67 | 16171.11 | 14765 | 58292.59 | 440 | 20.29 | 1 / 64 |
| S03-L / C64 | 338.89 | 21688.81 | 13147 | 69614.92 | N/A | N/A | 1 / 64 |
| S03-L / G64 | 1592.62 | 101927.90 | 13147 | 327159.69 | 440 | 20.25 | 1 / 64 |
| S03-L / G32 | 1999.31 | 127955.66 | 13147 | 410701.42 | 440 | 20.29 | 1 / 64 |
| W01-WR / C64 | 87.73 | 5614.46 | 14307 | 19610.87 | N/A | N/A | 1 / 64 |
| W01-WR / G64 | 523.21 | 33485.18 | 14307 | 116961.05 | 438 | 20.25 | 1 / 64 |
| W01-WR / G32 | 96.49 | 6175.18 | 14290 | 21543.77 | 440 | 20.29 | 1 / 64 |

### a800: numerical results against CPU FP64 high

CPU high/K=7 is a same-implementation regression witness, not independent physical truth. Strict wrong means abs error > max(3e-6,2e-6*abs(reference)) km/s on common finite ordinal slots. Budget wrong uses the same threshold for high and max(3e-6,2e-5*abs(reference)) for balanced. Missing/extra use one-to-one matches within the judged prefix, at that preset's budget. Rank shifts and unmatched close roots remain separate; matching coverage is retained in JSON. All counts below are per case, not independent totals across repeated B prefixes.

| Case / path | VALID / NOT_FOUND / UNRESOLVED / REJECTED | Strict / budget ordinal wrong | Set missing / extra | Rank shifts / cluster unmatched | Abs P99 / max km/s | Unjudged traces / prefix cells | Strict judged prefix / isolated budget |
|---|---|---|---|---|---|---|---|
| M01-R / C64 | 3 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.71e-07 / 4.72e-07 | 0 / 0 | pass / pass |
| M01-R / G64 | 3 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.71e-07 / 4.72e-07 | 0 / 0 | pass / pass |
| M01-R / G32 | 3 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 6.2e-07 / 6.2e-07 | 0 / 0 | pass / pass |
| M01-L / C64 | 2 / 3 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| M01-L / G64 | 2 / 3 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2.3e-11 / 2.32e-11 | 0 / 0 | pass / pass |
| M01-L / G32 | 2 / 3 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.35e-07 / 4.38e-07 | 0 / 0 | pass / pass |
| M02-R / C64 | 251 / 69 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.6e-07 / 1.19e-06 | 0 / 0 | pass / pass |
| M02-R / G64 | 251 / 69 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.6e-07 / 1.19e-06 | 0 / 0 | pass / pass |
| M02-R / G32 | 251 / 69 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.33e-06 / 1.77e-06 | 0 / 0 | pass / pass |
| M02-L / C64 | 220 / 100 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.08e-07 / 9.57e-07 | 0 / 0 | pass / pass |
| M02-L / G64 | 220 / 100 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.08e-07 / 9.57e-07 | 0 / 0 | pass / pass |
| M02-L / G32 | 220 / 100 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.52e-07 / 8.51e-07 | 0 / 0 | pass / pass |
| M03-R / C64 | 162 / 158 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.12e-06 / 1.21e-06 | 0 / 0 | pass / pass |
| M03-R / G64 | 162 / 158 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.12e-06 / 1.21e-06 | 0 / 0 | pass / pass |
| M03-R / G32 | 162 / 158 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.44e-06 / 2.45e-06 | 0 / 0 | pass / pass |
| M03-L / C64 | 109 / 211 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1e-06 / 1.11e-06 | 0 / 0 | pass / pass |
| M03-L / G64 | 109 / 211 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1e-06 / 1.11e-06 | 0 / 0 | pass / pass |
| M03-L / G32 | 109 / 211 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.67e-07 / 6.1e-07 | 0 / 0 | pass / pass |
| M04-R / C64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.01e-06 | 0 / 0 | pass / pass |
| M04-R / G64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.01e-06 | 0 / 0 | pass / pass |
| M04-R / G32 | 14765 / 5715 / 0 / 0 | 3 / 0 | 0 / 0 | 0 / 0 | 1.65e-06 / 4.15e-06 | 0 / 0 | fail / pass |
| M04-L / C64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.63e-07 / 1.87e-06 | 0 / 0 | pass / pass |
| M04-L / G64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.63e-07 / 1.87e-06 | 0 / 0 | pass / pass |
| M04-L / G32 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.6e-07 / 2.58e-06 | 0 / 0 | pass / pass |
| M05-R / C64 | 5215 / 5025 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.16e-06 / 1.86e-06 | 0 / 0 | pass / pass |
| M05-R / G64 | 5215 / 5025 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.16e-06 / 1.86e-06 | 0 / 0 | pass / pass |
| M05-R / G32 | 5215 / 5025 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.65e-06 / 2.66e-06 | 0 / 0 | pass / pass |
| M05-L / C64 | 3476 / 6764 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.1e-06 | 0 / 0 | pass / pass |
| M05-L / G64 | 3476 / 6764 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.1e-06 | 0 / 0 | pass / pass |
| M05-L / G32 | 3476 / 6764 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 6.69e-07 / 1.58e-06 | 0 / 0 | pass / pass |
| M06-R / C64 | 473743 / 181617 / 0 / 0 | 34 / 34 | 21 / 0 | 34 / 21 | 1.08e-06 / 0.904 | 0 / 0 | fail / pass |
| M06-R / G64 | 473743 / 181617 / 0 / 0 | 34 / 34 | 21 / 0 | 34 / 21 | 1.08e-06 / 0.904 | 0 / 0 | fail / pass |
| M06-R / G32 | 473773 / 181587 / 0 / 0 | 121 / 49 | 20 / 30 | 48 / 19 | 1.66e-06 / 2.25 | 0 / 0 | fail / fail |
| M06-L / C64 | 421616 / 233744 / 0 / 0 | 5 / 5 | 2 / 0 | 5 / 2 | 8.44e-07 / 0.179 | 0 / 0 | fail / pass |
| M06-L / G64 | 421616 / 233744 / 0 / 0 | 12 / 12 | 6 / 0 | 12 / 6 | 8.44e-07 / 0.179 | 0 / 0 | fail / pass |
| M06-L / G32 | 421618 / 233742 / 0 / 0 | 1 / 0 | 0 / 2 | 0 / 0 | 5.66e-07 / 3.29e-06 | 0 / 0 | fail / fail |
| S01-R / C64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.75e-07 / 1.54e-06 | 0 / 0 | pass / pass |
| S01-R / G64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.75e-07 / 1.54e-06 | 0 / 0 | pass / pass |
| S01-R / G32 | 4096 / 0 / 0 / 0 | 3 / 0 | 0 / 0 | 0 / 0 | 2.22e-06 / 4.15e-06 | 0 / 0 | fail / pass |
| S01-L / C64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8e-07 / 1.58e-06 | 0 / 0 | pass / pass |
| S01-L / G64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.01e-07 / 1.58e-06 | 0 / 0 | pass / pass |
| S01-L / G32 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.32e-07 / 2.58e-06 | 0 / 0 | pass / pass |
| S02-R / C64 | 18659 / 1821 / 0 / 0 | 3 / 3 | 2 / 0 | 3 / 2 | 1e-06 / 0.0551 | 0 / 0 | fail / pass |
| S02-R / G64 | 18659 / 1821 / 0 / 0 | 3 / 3 | 2 / 0 | 3 / 2 | 1e-06 / 0.0551 | 0 / 0 | fail / pass |
| S02-R / G32 | 18659 / 1821 / 0 / 0 | 10 / 3 | 2 / 0 | 3 / 2 | 1.49e-06 / 0.0551 | 0 / 0 | fail / pass |
| S02-L / C64 | 17274 / 3206 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.19e-07 / 1.7e-06 | 0 / 0 | pass / pass |
| S02-L / G64 | 17274 / 3206 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.19e-07 / 1.7e-06 | 0 / 0 | pass / pass |
| S02-L / G32 | 17274 / 3206 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.28e-07 / 8.57e-07 | 0 / 0 | pass / pass |
| S03-R / C64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| S03-R / G64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.63e-15 / 1.33e-07 | 0 / 0 | pass / pass |
| S03-R / G32 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.31e-06 / 3.67e-06 | 0 / 0 | pass / pass |
| S03-L / C64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| S03-L / G64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2.54e-11 / 3.58e-11 | 0 / 0 | pass / pass |
| S03-L / G32 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.58e-07 / 2.11e-06 | 0 / 0 | pass / pass |
| W01-WR / C64 | 14307 / 6173 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| W01-WR / G64 | 14307 / 6173 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.88e-16 / 1.12e-07 | 0 / 0 | pass / pass |
| W01-WR / G32 | 14290 / 6165 / 25 / 0 | 0 / 0 | 17 / 0 | 0 / 0 | 1.17e-06 / 3.12e-06 | 0 / 0 | fail / fail |

NOT_FOUND is not automatically a numerical failure or evidence of physical absence. Numerical failed cells are UNRESOLVED+REJECTED; their denominator is B*K*T. Failed traces have any such cell and use B*T; exceptions/OOM are separate failed-call counts. A budget pass remains approximate if cluster/rank differences exist. Unjudged domains are not counted as passes.

### a800: paired diagnostics

Each G64 row compares with C64 at the same preset; G32 compares with G64. Mismatch tuple is mask cells / status cells / capped-count traces / search-level traces / full-flag traces. Root counts cap at K+2 and include guards; equal saturated counts do not certify completeness. Flags are cumulative and precision-dependent. A successful fallback or attempted repair is not by itself a numerical failure.

| Case / path | Diagnostic mismatch tuple | Repair / shape fallback / Love phase fallback traces | Approximate / close-risk traces | Repeated output changes | Failed calls |
|---|---|---|---|---:|---:|
| M01-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 | 0 | 0 |
| M01-R / G32 | 0 / 0 / 0 / 0 / 1 | 0 / 0 / 0 | 1 / 1 | 0 | 0 |
| M01-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M01-L / G32 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M02-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 64 / 28 | 0 | 0 |
| M02-R / G32 | 0 / 0 / 0 / 0 / 11 | 0 / 0 / 0 | 64 / 39 | 0 | 0 |
| M02-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 7 | 7 / 7 | 0 | 0 |
| M02-L / G32 | 0 / 0 / 0 / 7 / 7 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M03-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 64 / 19 | 0 | 0 |
| M03-R / G32 | 0 / 0 / 0 / 0 / 39 | 0 / 0 / 0 | 64 / 58 | 0 | 0 |
| M03-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 17 | 17 / 9 | 0 | 0 |
| M03-L / G32 | 0 / 0 / 0 / 17 / 17 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M04-R / G64 | 0 / 0 / 0 / 0 / 7 | 0 / 0 / 0 | 4096 / 1313 | 0 | 0 |
| M04-R / G32 | 0 / 0 / 0 / 0 / 1052 | 0 / 0 / 0 | 4096 / 2349 | 0 | 0 |
| M04-L / G64 | 0 / 0 / 0 / 41 / 41 | 0 / 41 / 776 | 776 / 728 | 0 | 0 |
| M04-L / G32 | 0 / 0 / 0 / 780 / 780 | 0 / 0 / 6 | 6 / 21 | 0 | 0 |
| M05-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 2048 / 503 | 0 | 0 |
| M05-R / G32 | 0 / 0 / 0 / 0 / 1215 | 0 / 0 / 0 | 2048 / 1718 | 0 | 0 |
| M05-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 623 | 623 / 468 | 0 | 0 |
| M05-L / G32 | 0 / 0 / 0 / 621 / 621 | 0 / 0 / 4 | 4 / 4 | 0 | 0 |
| M06-R / G64 | 0 / 0 / 0 / 0 / 289 | 0 / 0 / 0 | 131072 / 43158 | 0 | 0 |
| M06-R / G32 | 30 / 30 / 16 / 2 / 34342 | 0 / 2 / 0 | 131072 / 77084 | 0 | 0 |
| M06-L / G64 | 0 / 0 / 0 / 1224 / 1231 | 0 / 1224 / 24333 | 24333 / 22568 | 0 | 0 |
| M06-L / G32 | 2 / 2 / 1 / 24389 / 24393 | 0 / 0 / 96 | 96 / 536 | 0 | 0 |
| S01-R / G64 | 0 / 0 / 0 / 0 / 12 | 0 / 0 / 0 | 4096 / 720 | 0 | 0 |
| S01-R / G32 | 0 / 0 / 0 / 0 / 1106 | 0 / 0 / 0 | 4096 / 1822 | 0 | 0 |
| S01-L / G64 | 0 / 0 / 0 / 24 / 35 | 0 / 24 / 776 | 776 / 591 | 0 | 0 |
| S01-L / G32 | 0 / 0 / 0 / 780 / 780 | 0 / 0 / 6 | 6 / 7 | 0 | 0 |
| S02-R / G64 | 0 / 0 / 0 / 0 / 5 | 0 / 0 / 0 | 4096 / 1801 | 0 | 0 |
| S02-R / G32 | 0 / 0 / 0 / 0 / 822 | 0 / 0 / 0 | 4096 / 2611 | 0 | 0 |
| S02-L / G64 | 0 / 0 / 0 / 46 / 46 | 0 / 46 / 828 | 828 / 755 | 0 | 0 |
| S02-L / G32 | 0 / 0 / 0 / 830 / 830 | 0 / 0 / 2 | 2 / 35 | 0 | 0 |
| S03-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 6 | 0 | 0 |
| S03-R / G32 | 0 / 0 / 0 / 0 / 593 | 20 / 0 / 0 | 0 / 599 | 0 | 0 |
| S03-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 776 | 0 / 15 | 0 | 0 |
| S03-L / G32 | 0 / 0 / 0 / 780 / 780 | 0 / 0 / 6 | 0 / 20 | 0 | 0 |
| W01-WR / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 9 | 0 | 0 |
| W01-WR / G32 | 17 / 25 / 5 / 216 / 2001 | 0 / 216 / 0 | 0 / 2007 | 0 | 0 |

Per-bit XOR, numerical_flags differences, status transitions via preserved arrays, reliability assessed/pass counts, first-call costs and 12 raw samples remain in the receipt/raw results. CPU/GPU bitwise identity is not an API guarantee.

### a800: velocity differences at the same preset

These isolate path/precision changes from the CPU-high preset comparison. Ordinal errors include rank shifts; no rematching is used to reduce their reported maxima. The matching counts here use the strict tolerance, even for balanced.

| Case / candidate → reference | Common / reference-valid cells | Strict ordinal wrong | Ordinal missing / extra | One-to-one missing / extra | Rank shifts | Abs P99 / max km/s | Relative P99 / max |
|---|---|---:|---|---|---:|---|---|
| M01-R / G64 → C64 | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 | 1.09e-16 / 1.11e-16 | 1.31e-16 / 1.34e-16 |
| M01-R / G32 → G64 | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 | 1.05e-06 / 1.06e-06 | 7.08e-07 / 7.09e-07 |
| M01-L / G64 → C64 | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 | 2.3e-11 / 2.32e-11 | 9.03e-12 / 9.12e-12 |
| M01-L / G32 → G64 | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 | 4.35e-07 / 4.38e-07 | 1.72e-07 / 1.72e-07 |
| M02-R / G64 → C64 | 251 / 251 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 3.55e-15 | 8.58e-16 / 1.61e-15 |
| M02-R / G32 → G64 | 251 / 251 | 0 | 0 / 0 | 0 / 0 | 0 | 1.92e-06 / 2.65e-06 | 9.08e-07 / 1.26e-06 |
| M02-L / G64 → C64 | 220 / 220 | 0 | 0 / 0 | 0 / 0 | 0 | 2.32e-11 / 2.32e-11 | 1.83e-11 / 2.25e-11 |
| M02-L / G32 → G64 | 220 / 220 | 0 | 0 / 0 | 0 / 0 | 0 | 6.88e-07 / 9.37e-07 | 4.54e-07 / 4.88e-07 |
| M03-R / G64 → C64 | 162 / 162 | 0 | 0 / 0 | 0 / 0 | 0 | 1.64e-15 / 4e-15 | 1.14e-15 / 2.7e-15 |
| M03-R / G32 → G64 | 162 / 162 | 0 | 0 / 0 | 0 / 0 | 0 | 2.33e-06 / 2.48e-06 | 1.12e-06 / 1.41e-06 |
| M03-L / G64 → C64 | 109 / 109 | 0 | 0 / 0 | 0 / 0 | 0 | 2.51e-11 / 2.54e-11 | 2.16e-11 / 2.25e-11 |
| M03-L / G32 → G64 | 109 / 109 | 0 | 0 / 0 | 0 / 0 | 0 | 1.39e-06 / 1.43e-06 | 4.8e-07 / 5.13e-07 |
| M04-R / G64 → C64 | 14765 / 14765 | 0 | 0 / 0 | 0 / 0 | 0 | 1.41e-15 / 7.99e-15 | 7.77e-16 / 3.08e-15 |
| M04-R / G32 → G64 | 14765 / 14765 | 5 | 0 / 0 | 5 / 5 | 0 | 2.27e-06 / 4.69e-06 | 9.41e-07 / 2.6e-06 |
| M04-L / G64 → C64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 5.86e-08 / 9.16e-07 | 6.64e-08 / 7.46e-07 |
| M04-L / G32 → G64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 9.74e-07 / 2.04e-06 | 4.51e-07 / 1.06e-06 |
| M05-R / G64 → C64 | 5215 / 5215 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 6.66e-15 | 1.24e-15 / 5.81e-15 |
| M05-R / G32 → G64 | 5215 / 5215 | 0 | 0 / 0 | 0 / 0 | 0 | 2.42e-06 / 3.28e-06 | 1.05e-06 / 2.96e-06 |
| M05-L / G64 → C64 | 3476 / 3476 | 0 | 0 / 0 | 0 / 0 | 0 | 2.55e-11 / 3.55e-11 | 2.56e-11 / 3.64e-11 |
| M05-L / G32 → G64 | 3476 / 3476 | 0 | 0 / 0 | 0 / 0 | 0 | 1.48e-06 / 2.04e-06 | 4.81e-07 / 1.22e-06 |
| M06-R / G64 → C64 | 473743 / 473743 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 5.58e-10 | 8.26e-16 / 1.95e-10 |
| M06-R / G32 → G64 | 473743 / 473743 | 104 | 0 / 30 | 80 / 107 | 24 | 2.27e-06 / 2.25 | 9.52e-07 / 0.678 |
| M06-L / G64 → C64 | 421616 / 421616 | 7 | 0 / 0 | 4 / 0 | 3 | 4.68e-08 / 0.128 | 5.53e-08 / 0.179 |
| M06-L / G32 → G64 | 421616 / 421616 | 13 | 0 / 2 | 6 / 8 | 7 | 9.83e-07 / 0.179 | 4.6e-07 / 0.197 |
| S01-R / G64 → C64 | 4096 / 4096 | 0 | 0 / 0 | 0 / 0 | 0 | 2.23e-15 / 5.33e-15 | 1.21e-15 / 3.08e-15 |
| S01-R / G32 → G64 | 4096 / 4096 | 5 | 0 / 0 | 5 / 4 | 0 | 2.6e-06 / 4.69e-06 | 1.24e-06 / 2.6e-06 |
| S01-L / G64 → C64 | 4096 / 4096 | 0 | 0 / 0 | 0 / 0 | 0 | 2.55e-11 / 9.16e-07 | 3.15e-11 / 7.46e-07 |
| S01-L / G32 → G64 | 4096 / 4096 | 0 | 0 / 0 | 0 / 0 | 0 | 9.75e-07 / 2.04e-06 | 4.76e-07 / 1.06e-06 |
| S02-R / G64 → C64 | 18659 / 18659 | 0 | 0 / 0 | 0 / 0 | 0 | 8.88e-16 / 1.51e-14 | 6.57e-16 / 3.02e-14 |
| S02-R / G32 → G64 | 18659 / 18659 | 7 | 0 / 0 | 7 / 7 | 0 | 2.19e-06 / 7.5e-06 | 9.8e-07 / 1.21e-05 |
| S02-L / G64 → C64 | 17274 / 17274 | 0 | 0 / 0 | 0 / 0 | 0 | 2.86e-08 / 5.94e-07 | 4.18e-08 / 8.87e-07 |
| S02-L / G32 → G64 | 17274 / 17274 | 0 | 0 / 0 | 0 / 0 | 0 | 8.49e-07 / 2.15e-06 | 4.67e-07 / 7.95e-07 |
| S03-R / G64 → C64 | 14765 / 14765 | 0 | 0 / 0 | 0 / 0 | 0 | 1.63e-15 / 1.33e-07 | 8.04e-16 / 6.51e-08 |
| S03-R / G32 → G64 | 14765 / 14765 | 0 | 0 / 0 | 0 / 0 | 0 | 1.31e-06 / 3.67e-06 | 6.35e-07 / 1.93e-06 |
| S03-L / G64 → C64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 2.54e-11 / 3.58e-11 | 2.19e-11 / 3.85e-11 |
| S03-L / G32 → G64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 5.58e-07 / 2.11e-06 | 2.24e-07 / 1.1e-06 |
| W01-WR / G64 → C64 | 14307 / 14307 | 0 | 0 / 0 | 0 / 0 | 0 | 8.88e-16 / 1.12e-07 | 6.65e-16 / 6.95e-08 |
| W01-WR / G32 → G64 | 14290 / 14307 | 0 | 17 / 0 | 17 / 0 | 0 | 1.17e-06 / 3.12e-06 | 5.66e-07 / 1.53e-06 |

### a800: independent checks

The 11 difficult Rayleigh cases retain the original expected root prefixes: eight witness files (reference_hi first five) and three B10000 files (reference first three, cmin=.2). Analytic Love and the factored low-velocity Rayleigh case are separate checks. Exact Love cutoff is unjudged. Wet Love is a physical invariance check, not new independent full-spectrum truth.

| Check | Path / preset | Reference-valid / common cells | Strict wrong | Missing / extra | UNRESOLVED / REJECTED cells | Repair traces |
|---|---|---|---:|---|---|---:|
| Rayleigh difficult (11) | C64 / high | 49 / 49 | 0 | 0 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G64 / high | 49 / 49 | 0 | 0 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G32 / high | 49 / 49 | 0 | 0 / 0 | 0 / 0 | 6 |
| Rayleigh difficult (11) | C64 / balanced | 49 / 49 | 24 | 13 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G64 / balanced | 49 / 49 | 24 | 13 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G32 / balanced | 49 / 49 | 24 | 13 / 0 | 0 / 0 | 0 |
| Analytic Love | C64 / high | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G64 / high | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G32 / high | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | C64 / balanced | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G64 / balanced | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G32 / balanced | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | C64 / high | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G64 / high | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G32 / high | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | C64 / balanced | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G64 / balanced | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G32 / balanced | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |

The S03-R G32 chunk_size=8 recheck had 0 strict ordinal differences, 0/0 missing/extra and 0 flag differences relative to the uncapped run. Its full stats and per-bit differences are retained. The high FP32 repair quota is per native chunk; changing chunking can affect the repair path even when velocities agree.

An additional **untimed wet-high follow-up** retains the original W01 timing/failure row. chunk_size=8: 32 attempted FP64 repairs, 25 UNRESOLVED cells, 17 missing reference roots; chunk_size=1: 39 attempted FP64 repairs, 25 UNRESOLVED cells, 17 missing reference roots. These diagnostic calls do not replace the original benchmark row.

## rtx5050: timing

Foreign GPU activity was observed in 1/19 workload groups. Host one-minute load ranged from 0.06 to 5.16. Rows marked † are diagnostic observations, not accepted uncontended speedups; unmarked rows retain their own monitoring record.

| Case | B × L × T; K / accuracy | CPU FP64 ms | CUDA FP64 ms | CUDA FP32 ms | GPU64 / CPU × | GPU32 / CPU × | FP32 / FP64 × |
|---|---|---:|---:|---:|---:|---:|---:|
| M01-R | 1 × 24 × 1; 5 / balanced | 0.956 [0.904,0.973] | 8.562 [8.543,8.578] | 2.874 [2.865,2.883] | 0.11 | 0.33 | 2.98 |
| M01-L | 1 × 24 × 1; 5 / balanced | 0.318 [0.293,0.337] | 6.737 [6.729,6.742] | 0.788 [0.765,0.792] | 0.05 | 0.40 | 8.55 |
| M02-R | 1 × 24 × 64; 5 / balanced | 6.743 [6.499,6.813] | 10.011 [9.914,10.024] | 3.701 [3.696,3.721] | 0.67 | 1.82 | 2.71 |
| M02-L | 1 × 24 × 64; 5 / balanced | 2.844 [2.798,2.888] | 15.956 [15.943,15.971] | 0.845 [0.836,0.856] | 0.18 | 3.36 | 18.88 |
| M03-R | 64 × 24 × 1; 5 / balanced | 6.969 [6.845,7.526] | 9.636 [9.621,9.660] | 6.855 [6.842,6.867] | 0.72 | 1.02 | 1.41 |
| M03-L | 64 × 24 × 1; 5 / balanced | 1.783 [1.694,1.883] | 12.410 [12.401,12.413] | 0.903 [0.892,0.909] | 0.14 | 1.97 | 13.75 |
| M04-R | 64 × 24 × 64; 5 / balanced | 378.699 [376.287,381.450] | 407.841 [407.126,408.442] | 74.352 [74.200,74.437] | 0.93 | 5.09 | 5.49 |
| M04-L | 64 × 24 × 64; 5 / balanced | 140.537 [140.304,141.956] | 358.997 [358.178,359.335] | 13.386 [13.364,13.404] | 0.39 | 10.50 | 26.82 |
| M05-R | 2048 × 24 × 1; 5 / balanced | 208.727 [207.923,210.262] | 217.078 [216.952,217.285] | 55.219 [55.127,55.623] | 0.96 | 3.78 | 3.93 |
| M05-L | 2048 × 24 × 1; 5 / balanced | 44.492 [44.219,44.648] | 183.480 [183.360,183.608] | 10.917 [10.813,10.972] | 0.24 | 4.08 | 16.81 |
| M06-R | 2048 × 24 × 64; 5 / balanced | 12150.122 [12139.468,12157.459] | 12920.877 [12919.102,12921.645] | 2177.995 [2177.144,2178.283] | 0.94 | 5.58 | 5.93 |
| M06-L† | 2048 × 24 × 64; 5 / balanced | 4523.843 [4522.134,4525.720] | 11023.783 [11023.037,11024.162] | 332.092 [331.533,332.526] | 0.41 | 13.62 | 33.20 |
| S01-R | 64 × 24 × 64; 1 / balanced | 289.343 [288.344,290.280] | 317.270 [316.671,317.785] | 60.639 [60.108,60.967] | 0.91 | 4.77 | 5.23 |
| S01-L | 64 × 24 × 64; 1 / balanced | 92.589 [89.642,93.385] | 341.940 [341.738,342.124] | 12.441 [12.405,12.492] | 0.27 | 7.44 | 27.49 |
| S02-R | 64 × 70 × 64; 5 / balanced | 1484.730 [1483.263,1486.130] | 1425.844 [1425.092,1429.075] | 141.279 [141.135,141.379] | 1.04 | 10.51 | 10.09 |
| S02-L | 64 × 70 × 64; 5 / balanced | 596.090 [594.614,598.292] | 1145.680 [1144.167,1147.349] | 36.450 [36.353,36.486] | 0.52 | 16.35 | 31.43 |
| S03-R | 64 × 24 × 64; 5 / high | 784.228 [782.105,784.750] | 1132.834 [1132.097,1134.220] | 549.751 [546.133,552.221] | 0.69 | 1.43 | 2.06 |
| S03-L | 64 × 24 × 64; 5 / high | 163.990 [161.739,166.372] | 427.568 [427.198,427.890] | 28.317 [28.252,28.378] | 0.38 | 5.79 | 15.10 |
| W01-WR | 64 × 24 × 64; 5 / high | 786.350 [783.992,787.705] | 1137.137 [1136.206,1138.827] | 731.018 [730.121,732.860] | 0.69 | 1.08 | 1.56 |

R=dry Rayleigh, L=Love, WR=wet Rayleigh. M01–M06 are the B/T grid, S01 is K=1, S02 is L=70 and S03 is high. Raw timing ratios do not imply equivalent root completeness. The numerical tables below retain mismatches and failed qualification.

### rtx5050: throughput and sampled memory

Rates use the steady median; traces/s=B*T/t and do not multiply by K. PID memory is measured separately with one CUDA worker, requested 10 ms NVML sampling. It includes process context overhead and is a sampled lower bound on instantaneous peak. Actual sampling intervals and native chunk capacities are in the receipt; native device bytes and workspace budgets are not total VRAM peaks.

Full paired retries were made for M06-L after monitoring flags. Each replacement requires identical numerical snapshot signatures on all three paths and an unflagged retry; samples are not pooled. Both attempts remain in JSON. The repeat also had an activity flag, so no replacement was accepted and the original flagged medians remain in the table. No competing compute process was resident in either group; the activity source was not retained, so the flag is not reclassified.

| Case / path | models/s | traces/s | VALID roots/call | valid roots/s | PID sampled peak MiB | Native device MiB | Chunks / capacity |
|---|---:|---:|---:|---:|---:|---:|---|
| M01-R / C64 | 1046.26 | 1046.26 | 3 | 3138.78 | N/A | N/A | 1 / 1 |
| M01-R / G64 | 116.80 | 116.80 | 3 | 350.39 | 92 | 0.0375 | 1 / 1 |
| M01-R / G32 | 347.99 | 347.99 | 3 | 1043.96 | 92 | 0.03806 | 1 / 1 |
| M01-L / C64 | 3146.42 | 3146.42 | 2 | 6292.84 | N/A | N/A | 1 / 1 |
| M01-L / G64 | 148.43 | 148.43 | 2 | 296.87 | 92 | 0.0375 | 1 / 1 |
| M01-L / G32 | 1268.89 | 1268.89 | 2 | 2537.78 | 92 | 0.03806 | 1 / 1 |
| M02-R / C64 | 148.31 | 9491.90 | 251 | 37226.03 | N/A | N/A | 1 / 1 |
| M02-R / G64 | 99.89 | 6392.89 | 251 | 25072.11 | 92 | 0.3172 | 1 / 1 |
| M02-R / G32 | 270.22 | 17294.21 | 251 | 67825.74 | 92 | 0.3178 | 1 / 1 |
| M02-L / C64 | 351.66 | 22506.17 | 220 | 77364.96 | N/A | N/A | 1 / 1 |
| M02-L / G64 | 62.67 | 4011.06 | 220 | 13788.02 | 92 | 0.3172 | 1 / 1 |
| M02-L / G32 | 1183.14 | 75721.19 | 220 | 260291.57 | 92 | 0.3178 | 1 / 1 |
| M03-R / C64 | 9184.05 | 9184.05 | 162 | 23247.14 | N/A | N/A | 1 / 64 |
| M03-R / G64 | 6641.43 | 6641.43 | 162 | 16811.12 | 94 | 2.38 | 1 / 64 |
| M03-R / G32 | 9336.49 | 9336.49 | 162 | 23633.00 | 94 | 2.416 | 1 / 64 |
| M03-L / C64 | 35899.08 | 35899.08 | 109 | 61140.62 | N/A | N/A | 1 / 64 |
| M03-L / G64 | 5157.06 | 5157.06 | 109 | 8783.13 | 94 | 2.38 | 1 / 64 |
| M03-L / G32 | 70885.85 | 70885.85 | 109 | 120727.47 | 94 | 2.416 | 1 / 64 |
| M04-R / C64 | 169.00 | 10815.99 | 14765 | 38988.79 | N/A | N/A | 1 / 64 |
| M04-R / G64 | 156.92 | 10043.13 | 14765 | 36202.82 | 112 | 20.25 | 1 / 64 |
| M04-R / G32 | 860.77 | 55089.14 | 14765 | 198581.81 | 112 | 20.29 | 1 / 64 |
| M04-L / C64 | 455.40 | 29145.35 | 13147 | 93548.31 | N/A | N/A | 1 / 64 |
| M04-L / G64 | 178.27 | 11409.58 | 13147 | 36621.52 | 112 | 20.25 | 1 / 64 |
| M04-L / G32 | 4781.13 | 305992.36 | 13147 | 982148.83 | 112 | 20.29 | 1 / 64 |
| M05-R / C64 | 9811.85 | 9811.85 | 5215 | 24984.77 | N/A | N/A | 1 / 2048 |
| M05-R / G64 | 9434.38 | 9434.38 | 5215 | 24023.57 | 168 | 76.16 | 1 / 2048 |
| M05-R / G32 | 37088.71 | 37088.71 | 5215 | 94442.21 | 168 | 77.29 | 1 / 2048 |
| M05-L / C64 | 46030.24 | 46030.24 | 3476 | 78125.54 | N/A | N/A | 1 / 2048 |
| M05-L / G64 | 11161.97 | 11161.97 | 3476 | 18944.84 | 168 | 76.16 | 1 / 2048 |
| M05-L / G32 | 187592.25 | 187592.25 | 3476 | 318393.87 | 168 | 77.29 | 1 / 2048 |
| M06-R / C64 | 168.56 | 10787.71 | 473743 | 38990.80 | N/A | N/A | 1 / 2048 |
| M06-R / G64 | 158.50 | 10144.20 | 473743 | 36664.92 | 594 | 490.8 | 2 / 1551 |
| M06-R / G32 | 940.31 | 60180.12 | 473773 | 217527.14 | 594 | 491 | 2 / 1549 |
| M06-L / C64 | 452.71 | 28973.60 | 421616 | 93198.64 | N/A | N/A | 1 / 2048 |
| M06-L / G64 | 185.78 | 11889.93 | 421616 | 38246.04 | 594 | 490.8 | 2 / 1551 |
| M06-L / G32 | 6166.97 | 394686.29 | 421618 | 1269583.46 | 594 | 491 | 2 / 1549 |
| S01-R / C64 | 221.19 | 14156.22 | 4096 | 14156.22 | N/A | N/A | 1 / 64 |
| S01-R / G64 | 201.72 | 12910.15 | 4096 | 12910.15 | 110 | 19.13 | 1 / 64 |
| S01-R / G32 | 1055.43 | 67547.45 | 4096 | 67547.45 | 110 | 19.16 | 1 / 64 |
| S01-L / C64 | 691.22 | 44238.36 | 4096 | 44238.36 | N/A | N/A | 1 / 64 |
| S01-L / G64 | 187.17 | 11978.71 | 4096 | 11978.71 | 110 | 19.13 | 1 / 64 |
| S01-L / G32 | 5144.33 | 329237.33 | 4096 | 329237.33 | 110 | 19.16 | 1 / 64 |
| S02-R / C64 | 43.11 | 2758.75 | 18659 | 12567.27 | N/A | N/A | 1 / 64 |
| S02-R / G64 | 44.89 | 2872.68 | 18659 | 13086.28 | 112 | 20.43 | 1 / 64 |
| S02-R / G32 | 453.00 | 28992.27 | 18659 | 132071.98 | 112 | 20.54 | 1 / 64 |
| S02-L / C64 | 107.37 | 6871.44 | 17274 | 28978.83 | N/A | N/A | 1 / 64 |
| S02-L / G64 | 55.86 | 3575.17 | 17274 | 15077.51 | 112 | 20.43 | 1 / 64 |
| S02-L / G32 | 1755.81 | 112372.12 | 17274 | 473905.28 | 112 | 20.54 | 1 / 64 |
| S03-R / C64 | 81.61 | 5222.97 | 14765 | 18827.42 | N/A | N/A | 1 / 64 |
| S03-R / G64 | 56.50 | 3615.71 | 14765 | 13033.68 | 112 | 20.25 | 1 / 64 |
| S03-R / G32 | 116.42 | 7450.64 | 14765 | 26857.61 | 112 | 20.29 | 1 / 64 |
| S03-L / C64 | 390.27 | 24977.20 | 13147 | 80169.75 | N/A | N/A | 1 / 64 |
| S03-L / G64 | 149.68 | 9579.76 | 13147 | 30748.31 | 112 | 20.25 | 1 / 64 |
| S03-L / G32 | 2260.15 | 144649.34 | 13147 | 464283.40 | 112 | 20.29 | 1 / 64 |
| W01-WR / C64 | 81.39 | 5208.88 | 14307 | 18194.20 | N/A | N/A | 1 / 64 |
| W01-WR / G64 | 56.28 | 3602.03 | 14307 | 12581.60 | 112 | 20.25 | 1 / 64 |
| W01-WR / G32 | 87.55 | 5603.15 | 14287 | 19543.99 | 112 | 20.29 | 1 / 64 |

### rtx5050: numerical results against CPU FP64 high

CPU high/K=7 is a same-implementation regression witness, not independent physical truth. Strict wrong means abs error > max(3e-6,2e-6*abs(reference)) km/s on common finite ordinal slots. Budget wrong uses the same threshold for high and max(3e-6,2e-5*abs(reference)) for balanced. Missing/extra use one-to-one matches within the judged prefix, at that preset's budget. Rank shifts and unmatched close roots remain separate; matching coverage is retained in JSON. All counts below are per case, not independent totals across repeated B prefixes.

| Case / path | VALID / NOT_FOUND / UNRESOLVED / REJECTED | Strict / budget ordinal wrong | Set missing / extra | Rank shifts / cluster unmatched | Abs P99 / max km/s | Unjudged traces / prefix cells | Strict judged prefix / isolated budget |
|---|---|---|---|---|---|---|---|
| M01-R / C64 | 3 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.71e-07 / 4.72e-07 | 0 / 0 | pass / pass |
| M01-R / G64 | 3 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.71e-07 / 4.72e-07 | 0 / 0 | pass / pass |
| M01-R / G32 | 3 / 2 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 6.2e-07 / 6.2e-07 | 0 / 0 | pass / pass |
| M01-L / C64 | 2 / 3 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| M01-L / G64 | 2 / 3 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2.3e-11 / 2.32e-11 | 0 / 0 | pass / pass |
| M01-L / G32 | 2 / 3 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 4.35e-07 / 4.38e-07 | 0 / 0 | pass / pass |
| M02-R / C64 | 251 / 69 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.6e-07 / 1.19e-06 | 0 / 0 | pass / pass |
| M02-R / G64 | 251 / 69 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.6e-07 / 1.19e-06 | 0 / 0 | pass / pass |
| M02-R / G32 | 251 / 69 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.33e-06 / 1.77e-06 | 0 / 0 | pass / pass |
| M02-L / C64 | 220 / 100 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.08e-07 / 9.57e-07 | 0 / 0 | pass / pass |
| M02-L / G64 | 220 / 100 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.08e-07 / 9.57e-07 | 0 / 0 | pass / pass |
| M02-L / G32 | 220 / 100 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.52e-07 / 8.51e-07 | 0 / 0 | pass / pass |
| M03-R / C64 | 162 / 158 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.12e-06 / 1.21e-06 | 0 / 0 | pass / pass |
| M03-R / G64 | 162 / 158 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.12e-06 / 1.21e-06 | 0 / 0 | pass / pass |
| M03-R / G32 | 162 / 158 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.44e-06 / 2.45e-06 | 0 / 0 | pass / pass |
| M03-L / C64 | 109 / 211 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1e-06 / 1.11e-06 | 0 / 0 | pass / pass |
| M03-L / G64 | 109 / 211 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1e-06 / 1.11e-06 | 0 / 0 | pass / pass |
| M03-L / G32 | 109 / 211 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.67e-07 / 6.1e-07 | 0 / 0 | pass / pass |
| M04-R / C64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.01e-06 | 0 / 0 | pass / pass |
| M04-R / G64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.01e-06 | 0 / 0 | pass / pass |
| M04-R / G32 | 14765 / 5715 / 0 / 0 | 3 / 0 | 0 / 0 | 0 / 0 | 1.65e-06 / 4.15e-06 | 0 / 0 | fail / pass |
| M04-L / C64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.63e-07 / 1.87e-06 | 0 / 0 | pass / pass |
| M04-L / G64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.63e-07 / 1.87e-06 | 0 / 0 | pass / pass |
| M04-L / G32 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.6e-07 / 2.58e-06 | 0 / 0 | pass / pass |
| M05-R / C64 | 5215 / 5025 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.16e-06 / 1.86e-06 | 0 / 0 | pass / pass |
| M05-R / G64 | 5215 / 5025 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.16e-06 / 1.86e-06 | 0 / 0 | pass / pass |
| M05-R / G32 | 5215 / 5025 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.65e-06 / 2.66e-06 | 0 / 0 | pass / pass |
| M05-L / C64 | 3476 / 6764 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.1e-06 | 0 / 0 | pass / pass |
| M05-L / G64 | 3476 / 6764 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.08e-06 / 2.1e-06 | 0 / 0 | pass / pass |
| M05-L / G32 | 3476 / 6764 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 6.69e-07 / 1.58e-06 | 0 / 0 | pass / pass |
| M06-R / C64 | 473743 / 181617 / 0 / 0 | 34 / 34 | 21 / 0 | 34 / 21 | 1.08e-06 / 0.904 | 0 / 0 | fail / pass |
| M06-R / G64 | 473743 / 181617 / 0 / 0 | 34 / 34 | 21 / 0 | 34 / 21 | 1.08e-06 / 0.904 | 0 / 0 | fail / pass |
| M06-R / G32 | 473773 / 181587 / 0 / 0 | 121 / 49 | 20 / 30 | 48 / 19 | 1.66e-06 / 2.25 | 0 / 0 | fail / fail |
| M06-L / C64 | 421616 / 233744 / 0 / 0 | 5 / 5 | 2 / 0 | 5 / 2 | 8.44e-07 / 0.179 | 0 / 0 | fail / pass |
| M06-L / G64 | 421616 / 233744 / 0 / 0 | 12 / 12 | 6 / 0 | 12 / 6 | 8.44e-07 / 0.179 | 0 / 0 | fail / pass |
| M06-L / G32 | 421618 / 233742 / 0 / 0 | 1 / 0 | 0 / 2 | 0 / 0 | 5.66e-07 / 3.29e-06 | 0 / 0 | fail / fail |
| S01-R / C64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.75e-07 / 1.54e-06 | 0 / 0 | pass / pass |
| S01-R / G64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 9.75e-07 / 1.54e-06 | 0 / 0 | pass / pass |
| S01-R / G32 | 4096 / 0 / 0 / 0 | 3 / 0 | 0 / 0 | 0 / 0 | 2.22e-06 / 4.15e-06 | 0 / 0 | fail / pass |
| S01-L / C64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8e-07 / 1.58e-06 | 0 / 0 | pass / pass |
| S01-L / G64 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.01e-07 / 1.58e-06 | 0 / 0 | pass / pass |
| S01-L / G32 | 4096 / 0 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.32e-07 / 2.58e-06 | 0 / 0 | pass / pass |
| S02-R / C64 | 18659 / 1821 / 0 / 0 | 3 / 3 | 2 / 0 | 3 / 2 | 1e-06 / 0.0551 | 0 / 0 | fail / pass |
| S02-R / G64 | 18659 / 1821 / 0 / 0 | 3 / 3 | 2 / 0 | 3 / 2 | 1e-06 / 0.0551 | 0 / 0 | fail / pass |
| S02-R / G32 | 18659 / 1821 / 0 / 0 | 10 / 3 | 2 / 0 | 3 / 2 | 1.49e-06 / 0.0551 | 0 / 0 | fail / pass |
| S02-L / C64 | 17274 / 3206 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.19e-07 / 1.7e-06 | 0 / 0 | pass / pass |
| S02-L / G64 | 17274 / 3206 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 7.19e-07 / 1.7e-06 | 0 / 0 | pass / pass |
| S02-L / G32 | 17274 / 3206 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.28e-07 / 8.57e-07 | 0 / 0 | pass / pass |
| S03-R / C64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| S03-R / G64 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.78e-15 / 1.33e-07 | 0 / 0 | pass / pass |
| S03-R / G32 | 14765 / 5715 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1.31e-06 / 3.67e-06 | 0 / 0 | pass / pass |
| S03-L / C64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| S03-L / G64 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 2.54e-11 / 3.58e-11 | 0 / 0 | pass / pass |
| S03-L / G32 | 13147 / 7333 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 5.58e-07 / 2.11e-06 | 0 / 0 | pass / pass |
| W01-WR / C64 | 14307 / 6173 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | pass / pass |
| W01-WR / G64 | 14307 / 6173 / 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 8.88e-16 / 1.12e-07 | 0 / 0 | pass / pass |
| W01-WR / G32 | 14287 / 6163 / 30 / 0 | 0 / 0 | 20 / 0 | 0 / 0 | 1.17e-06 / 3.12e-06 | 0 / 0 | fail / fail |

NOT_FOUND is not automatically a numerical failure or evidence of physical absence. Numerical failed cells are UNRESOLVED+REJECTED; their denominator is B*K*T. Failed traces have any such cell and use B*T; exceptions/OOM are separate failed-call counts. A budget pass remains approximate if cluster/rank differences exist. Unjudged domains are not counted as passes.

### rtx5050: paired diagnostics

Each G64 row compares with C64 at the same preset; G32 compares with G64. Mismatch tuple is mask cells / status cells / capped-count traces / search-level traces / full-flag traces. Root counts cap at K+2 and include guards; equal saturated counts do not certify completeness. Flags are cumulative and precision-dependent. A successful fallback or attempted repair is not by itself a numerical failure.

| Case / path | Diagnostic mismatch tuple | Repair / shape fallback / Love phase fallback traces | Approximate / close-risk traces | Repeated output changes | Failed calls |
|---|---|---|---|---:|---:|
| M01-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 | 0 | 0 |
| M01-R / G32 | 0 / 0 / 0 / 0 / 1 | 0 / 0 / 0 | 1 / 1 | 0 | 0 |
| M01-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M01-L / G32 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M02-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 64 / 28 | 0 | 0 |
| M02-R / G32 | 0 / 0 / 0 / 0 / 11 | 0 / 0 / 0 | 64 / 39 | 0 | 0 |
| M02-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 7 | 7 / 7 | 0 | 0 |
| M02-L / G32 | 0 / 0 / 0 / 7 / 7 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M03-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 64 / 19 | 0 | 0 |
| M03-R / G32 | 0 / 0 / 0 / 0 / 39 | 0 / 0 / 0 | 64 / 58 | 0 | 0 |
| M03-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 17 | 17 / 9 | 0 | 0 |
| M03-L / G32 | 0 / 0 / 0 / 17 / 17 | 0 / 0 / 0 | 0 / 0 | 0 | 0 |
| M04-R / G64 | 0 / 0 / 0 / 0 / 7 | 0 / 0 / 0 | 4096 / 1313 | 0 | 0 |
| M04-R / G32 | 0 / 0 / 0 / 0 / 1053 | 0 / 0 / 0 | 4096 / 2350 | 0 | 0 |
| M04-L / G64 | 0 / 0 / 0 / 41 / 41 | 0 / 41 / 776 | 776 / 728 | 0 | 0 |
| M04-L / G32 | 0 / 0 / 0 / 780 / 780 | 0 / 0 / 6 | 6 / 21 | 0 | 0 |
| M05-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 2048 / 503 | 0 | 0 |
| M05-R / G32 | 0 / 0 / 0 / 0 / 1218 | 0 / 0 / 0 | 2048 / 1721 | 0 | 0 |
| M05-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 623 | 623 / 468 | 0 | 0 |
| M05-L / G32 | 0 / 0 / 0 / 621 / 621 | 0 / 0 / 4 | 4 / 4 | 0 | 0 |
| M06-R / G64 | 0 / 0 / 0 / 0 / 289 | 0 / 0 / 0 | 131072 / 43158 | 0 | 0 |
| M06-R / G32 | 30 / 30 / 16 / 2 / 34340 | 0 / 2 / 0 | 131072 / 77082 | 0 | 0 |
| M06-L / G64 | 0 / 0 / 0 / 1224 / 1231 | 0 / 1224 / 24333 | 24333 / 22568 | 0 | 0 |
| M06-L / G32 | 2 / 2 / 1 / 24389 / 24393 | 0 / 0 / 96 | 96 / 536 | 0 | 0 |
| S01-R / G64 | 0 / 0 / 0 / 0 / 12 | 0 / 0 / 0 | 4096 / 720 | 0 | 0 |
| S01-R / G32 | 0 / 0 / 0 / 0 / 1107 | 0 / 0 / 0 | 4096 / 1823 | 0 | 0 |
| S01-L / G64 | 0 / 0 / 0 / 24 / 35 | 0 / 24 / 776 | 776 / 591 | 0 | 0 |
| S01-L / G32 | 0 / 0 / 0 / 780 / 780 | 0 / 0 / 6 | 6 / 7 | 0 | 0 |
| S02-R / G64 | 0 / 0 / 0 / 0 / 5 | 0 / 0 / 0 | 4096 / 1801 | 0 | 0 |
| S02-R / G32 | 0 / 0 / 0 / 0 / 825 | 0 / 0 / 0 | 4096 / 2614 | 0 | 0 |
| S02-L / G64 | 0 / 0 / 0 / 46 / 46 | 0 / 46 / 828 | 828 / 755 | 0 | 0 |
| S02-L / G32 | 0 / 0 / 0 / 830 / 830 | 0 / 0 / 2 | 2 / 35 | 0 | 0 |
| S03-R / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 6 | 0 | 0 |
| S03-R / G32 | 0 / 0 / 0 / 0 / 593 | 20 / 0 / 0 | 0 / 599 | 0 | 0 |
| S03-L / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 776 | 0 / 15 | 0 | 0 |
| S03-L / G32 | 0 / 0 / 0 / 780 / 780 | 0 / 0 / 6 | 0 / 20 | 0 | 0 |
| W01-WR / G64 | 0 / 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 9 | 0 | 0 |
| W01-WR / G32 | 20 / 30 / 6 / 216 / 2001 | 0 / 216 / 0 | 0 / 2007 | 0 | 0 |

Per-bit XOR, numerical_flags differences, status transitions via preserved arrays, reliability assessed/pass counts, first-call costs and 12 raw samples remain in the receipt/raw results. CPU/GPU bitwise identity is not an API guarantee.

### rtx5050: velocity differences at the same preset

These isolate path/precision changes from the CPU-high preset comparison. Ordinal errors include rank shifts; no rematching is used to reduce their reported maxima. The matching counts here use the strict tolerance, even for balanced.

| Case / candidate → reference | Common / reference-valid cells | Strict ordinal wrong | Ordinal missing / extra | One-to-one missing / extra | Rank shifts | Abs P99 / max km/s | Relative P99 / max |
|---|---|---:|---|---|---:|---|---|
| M01-R / G64 → C64 | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 | 1.09e-16 / 1.11e-16 | 1.31e-16 / 1.34e-16 |
| M01-R / G32 → G64 | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 | 1.05e-06 / 1.06e-06 | 7.08e-07 / 7.09e-07 |
| M01-L / G64 → C64 | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 | 2.3e-11 / 2.32e-11 | 9.03e-12 / 9.12e-12 |
| M01-L / G32 → G64 | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 | 4.35e-07 / 4.38e-07 | 1.72e-07 / 1.72e-07 |
| M02-R / G64 → C64 | 251 / 251 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 3.55e-15 | 8.58e-16 / 1.61e-15 |
| M02-R / G32 → G64 | 251 / 251 | 0 | 0 / 0 | 0 / 0 | 0 | 1.92e-06 / 2.65e-06 | 9.08e-07 / 1.26e-06 |
| M02-L / G64 → C64 | 220 / 220 | 0 | 0 / 0 | 0 / 0 | 0 | 2.32e-11 / 2.32e-11 | 1.83e-11 / 2.25e-11 |
| M02-L / G32 → G64 | 220 / 220 | 0 | 0 / 0 | 0 / 0 | 0 | 6.88e-07 / 9.37e-07 | 4.54e-07 / 4.88e-07 |
| M03-R / G64 → C64 | 162 / 162 | 0 | 0 / 0 | 0 / 0 | 0 | 1.64e-15 / 4e-15 | 1.14e-15 / 2.7e-15 |
| M03-R / G32 → G64 | 162 / 162 | 0 | 0 / 0 | 0 / 0 | 0 | 2.33e-06 / 2.48e-06 | 1.12e-06 / 1.41e-06 |
| M03-L / G64 → C64 | 109 / 109 | 0 | 0 / 0 | 0 / 0 | 0 | 2.51e-11 / 2.54e-11 | 2.16e-11 / 2.25e-11 |
| M03-L / G32 → G64 | 109 / 109 | 0 | 0 / 0 | 0 / 0 | 0 | 1.39e-06 / 1.43e-06 | 4.8e-07 / 5.13e-07 |
| M04-R / G64 → C64 | 14765 / 14765 | 0 | 0 / 0 | 0 / 0 | 0 | 1.55e-15 / 7.99e-15 | 7.61e-16 / 3.08e-15 |
| M04-R / G32 → G64 | 14765 / 14765 | 5 | 0 / 0 | 5 / 5 | 0 | 2.27e-06 / 4.69e-06 | 9.41e-07 / 2.6e-06 |
| M04-L / G64 → C64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 5.86e-08 / 9.16e-07 | 6.64e-08 / 7.46e-07 |
| M04-L / G32 → G64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 9.74e-07 / 2.04e-06 | 4.51e-07 / 1.06e-06 |
| M05-R / G64 → C64 | 5215 / 5215 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 6.66e-15 | 1.23e-15 / 5.81e-15 |
| M05-R / G32 → G64 | 5215 / 5215 | 0 | 0 / 0 | 0 / 0 | 0 | 2.42e-06 / 3.28e-06 | 1.05e-06 / 2.96e-06 |
| M05-L / G64 → C64 | 3476 / 3476 | 0 | 0 / 0 | 0 / 0 | 0 | 2.55e-11 / 3.55e-11 | 2.56e-11 / 3.64e-11 |
| M05-L / G32 → G64 | 3476 / 3476 | 0 | 0 / 0 | 0 / 0 | 0 | 1.48e-06 / 2.04e-06 | 4.81e-07 / 1.22e-06 |
| M06-R / G64 → C64 | 473743 / 473743 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 5.58e-10 | 8.26e-16 / 1.95e-10 |
| M06-R / G32 → G64 | 473743 / 473743 | 104 | 0 / 30 | 80 / 107 | 24 | 2.27e-06 / 2.25 | 9.52e-07 / 0.678 |
| M06-L / G64 → C64 | 421616 / 421616 | 7 | 0 / 0 | 4 / 0 | 3 | 4.68e-08 / 0.128 | 5.53e-08 / 0.179 |
| M06-L / G32 → G64 | 421616 / 421616 | 13 | 0 / 2 | 6 / 8 | 7 | 9.83e-07 / 0.179 | 4.6e-07 / 0.197 |
| S01-R / G64 → C64 | 4096 / 4096 | 0 | 0 / 0 | 0 / 0 | 0 | 2.45e-15 / 5.33e-15 | 1.21e-15 / 3.08e-15 |
| S01-R / G32 → G64 | 4096 / 4096 | 5 | 0 / 0 | 5 / 4 | 0 | 2.6e-06 / 4.69e-06 | 1.24e-06 / 2.6e-06 |
| S01-L / G64 → C64 | 4096 / 4096 | 0 | 0 / 0 | 0 / 0 | 0 | 2.55e-11 / 9.16e-07 | 3.15e-11 / 7.46e-07 |
| S01-L / G32 → G64 | 4096 / 4096 | 0 | 0 / 0 | 0 / 0 | 0 | 9.75e-07 / 2.04e-06 | 4.76e-07 / 1.06e-06 |
| S02-R / G64 → C64 | 18659 / 18659 | 0 | 0 / 0 | 0 / 0 | 0 | 8.88e-16 / 1.51e-14 | 6.5e-16 / 3.02e-14 |
| S02-R / G32 → G64 | 18659 / 18659 | 7 | 0 / 0 | 7 / 7 | 0 | 2.19e-06 / 7.5e-06 | 9.8e-07 / 1.21e-05 |
| S02-L / G64 → C64 | 17274 / 17274 | 0 | 0 / 0 | 0 / 0 | 0 | 2.86e-08 / 5.94e-07 | 4.18e-08 / 8.87e-07 |
| S02-L / G32 → G64 | 17274 / 17274 | 0 | 0 / 0 | 0 / 0 | 0 | 8.49e-07 / 2.15e-06 | 4.67e-07 / 7.95e-07 |
| S03-R / G64 → C64 | 14765 / 14765 | 0 | 0 / 0 | 0 / 0 | 0 | 1.78e-15 / 1.33e-07 | 7.97e-16 / 6.51e-08 |
| S03-R / G32 → G64 | 14765 / 14765 | 0 | 0 / 0 | 0 / 0 | 0 | 1.31e-06 / 3.67e-06 | 6.35e-07 / 1.93e-06 |
| S03-L / G64 → C64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 2.54e-11 / 3.58e-11 | 2.2e-11 / 3.85e-11 |
| S03-L / G32 → G64 | 13147 / 13147 | 0 | 0 / 0 | 0 / 0 | 0 | 5.58e-07 / 2.11e-06 | 2.24e-07 / 1.1e-06 |
| W01-WR / G64 → C64 | 14307 / 14307 | 0 | 0 / 0 | 0 / 0 | 0 | 8.88e-16 / 1.12e-07 | 6.62e-16 / 6.95e-08 |
| W01-WR / G32 → G64 | 14287 / 14307 | 0 | 20 / 0 | 20 / 0 | 0 | 1.17e-06 / 3.12e-06 | 5.66e-07 / 1.53e-06 |

### rtx5050: independent checks

The 11 difficult Rayleigh cases retain the original expected root prefixes: eight witness files (reference_hi first five) and three B10000 files (reference first three, cmin=.2). Analytic Love and the factored low-velocity Rayleigh case are separate checks. Exact Love cutoff is unjudged. Wet Love is a physical invariance check, not new independent full-spectrum truth.

| Check | Path / preset | Reference-valid / common cells | Strict wrong | Missing / extra | UNRESOLVED / REJECTED cells | Repair traces |
|---|---|---|---:|---|---|---:|
| Rayleigh difficult (11) | C64 / high | 49 / 49 | 0 | 0 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G64 / high | 49 / 49 | 0 | 0 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G32 / high | 49 / 49 | 0 | 0 / 0 | 0 / 0 | 6 |
| Rayleigh difficult (11) | C64 / balanced | 49 / 49 | 24 | 13 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G64 / balanced | 49 / 49 | 24 | 13 / 0 | 0 / 0 | 0 |
| Rayleigh difficult (11) | G32 / balanced | 49 / 49 | 24 | 13 / 0 | 0 / 0 | 0 |
| Analytic Love | C64 / high | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G64 / high | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G32 / high | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | C64 / balanced | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G64 / balanced | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Analytic Love | G32 / balanced | 3 / 3 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | C64 / high | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G64 / high | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G32 / high | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | C64 / balanced | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G64 / balanced | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |
| Low-velocity Rayleigh | G32 / balanced | 2 / 2 | 0 | 0 / 0 | 0 / 0 | 0 |

The S03-R G32 chunk_size=8 recheck had 0 strict ordinal differences, 0/0 missing/extra and 0 flag differences relative to the uncapped run. Its full stats and per-bit differences are retained. The high FP32 repair quota is per native chunk; changing chunking can affect the repair path even when velocities agree.

An additional **untimed wet-high follow-up** retains the original W01 timing/failure row. chunk_size=8: 32 attempted FP64 repairs, 30 UNRESOLVED cells, 20 missing reference roots; chunk_size=1: 39 attempted FP64 repairs, 30 UNRESOLVED cells, 20 missing reference roots. These diagnostic calls do not replace the original benchmark row.

## Reproduction

Install the matching released 0.4.0 CPU/CUDA wheels for each host; the two runs used NumPy 2.2.6. Install scipy and nvidia-ml-py for the auxiliary validation and monitoring, using the versions recorded per platform. Invoke scripts with the selected environment's Python; the source shebang is the maintainer's local convention.

```sh
python tools/benchmark_backends.py prepare --output /new/inputs
# Copy the frozen input directory unchanged to each host.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=GPU-UUID python tools/benchmark_backends.py run \
  --input /new/inputs --output /new/run --gpu-uuid GPU-UUID \
  --threads 8 --affinity 0,1,2,3,4,5,6,7
# A800: use 32 physical cores on the GPU-local NUMA node instead.
python tools/report_backends.py --run a800=/path/to/a800/run \
  --run rtx5050=/path/to/rtx5050/run --output docs
```

A completed group is checkpointed. Reusing a run directory requires the same input manifest, tool hash, threads, affinity and GPU. A resumed run retains per-group dates and samples; it does not fabricate a single uninterrupted end-to-end pass. Memory runs and profile=True calls are separate from the main timing. Complete per-call NPZ outputs and monitor logs stay outside the source tree; the public receipts retain result hashes, scalar counters, all raw steady samples, numerical accounting and stage profiles.

The matrix tests backends/precision within 0.4.0. It does not measure a version-to-version speedup or certify physical branch identity.

## Cross-platform numerical consistency

Both hosts used byte-identical frozen inputs. This comparison changes GPU architecture, CPU, Python ABI and CUDA/native builds together; neither host is independent truth. All 57 comparisons are in the [cross-platform receipt](benchmarks/backend-0.4.0-20261112-cross-platform.json). Selected rows compare the same path and preset:

| Case / path | Strict ordinal differences | Mask / status / root-count mismatches | UNRESOLVED cells A800 / RTX 5050 | Max common abs difference km/s |
|---|---:|---|---|---:|
| M06-L / G32 | 0 | 0 / 0 / 0 | 0 / 0 | 0 |
| M06-L / G64 | 0 | 0 / 0 / 0 | 0 / 0 | 0 |
| M06-R / G32 | 0 | 0 / 0 / 0 | 0 / 0 | 0 |
| M06-R / G64 | 0 | 0 / 0 / 0 | 0 / 0 | 4.44e-15 |
| S03-R / G32 | 0 | 0 / 0 / 0 | 0 / 0 | 0 |
| S03-R / G64 | 0 | 0 / 0 / 0 | 0 / 0 | 1.78e-15 |
| W01-WR / G32 | 0 | 3 / 5 / 1 | 25 / 30 | 2.38e-07 |
| W01-WR / G64 | 0 | 0 / 0 / 0 | 0 / 0 | 5.51e-08 |
