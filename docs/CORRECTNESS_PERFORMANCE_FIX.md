# 严格精度修复实验归档

本文件记录此前的严格精度方案，不代表当前 balanced/fast 行为。按用户后续指定的 0.5% 观测容差，当前方案见 [TOLERANT_PRESETS.md](TOLERANT_PRESETS.md)。

## 0.3.0 求根正确性修复与性能实验

本次修改基于 `005f693`，尚未发布新版本。三个精度档位均恢复完整 high 搜索及独立验证；CPU 通过精确计算复用获得约 **1.31×** 加速，CUDA 保持 high 基线性能。原版 0.3.0 的约 2.4× 加速来自已撤除的搜索简化，不适用于修复后代码。

## 修复了什么

1. 删除基于已观测根间距的提前接受和跳过独立验证。两个稀疏网格可能同时漏掉近根，不能以它们的一致性证明根集完整。`witness_model7584_period9`、`witness_model9630_period2` 之前在 balanced/fast 下静默错排模态，最大偏差分别约 0.0611、0.0732 km/s，现在恢复正确前三根。
2. 所有档位恢复 32,768/65,536 的 dense 网格下限与完整验证，删除绕过安全检查的低密度接受路径。`witness_model9384_period11` 之前由 UNRESOLVED 变成错误 VALID，最大偏差约 0.20591 km/s；现在保留原有 UNRESOLVED。
3. 为缩小 CPU dense 扫描块增加精确零点边界处理。构造的 Love 根恰好落在第一个 64 点块边界时，扫描额外读取右邻点，避免把合法根当成非有限失败。
4. 同步 CUDA 插件的 12 个本地源文件，并让开发 checkout 优先使用主 `cpp/` 树，防止陈旧的插件副本覆盖修复。独立插件 sdist 使用同步后的自包含源码。

`adaptive_period_threshold` 保留调用兼容性，但不再改变搜索，因为提前接受已关闭。`search_level` 当前仅在 3/5 接受；所有 early/skip/coarse/trough 计数均为零。`high` 保留原始执行路径；`balanced`/`fast` 使用相同验证强度，名称不再表示放宽接受精度。

## 保留的优化

- **精确 bracket 缓存**：同一 model-period 内，完全相同的 bracket 直接复用已完成的 FP64 精修结果；不合并相邻根。
- **采样缓存并集**：在 2048 点上限内，保留各采样层和验证族中已有的精确速度键。
- **嵌套 dense 复用**：细网格的偶数点仅在 FP64 速度坐标完全相等时复用粗网格函数值；最多缓存 32769 个值。
- **CPU prefix tiles**：balanced/fast 分别使用 64/32 点块，减少找到 `nmodes+2` 个 guarded 根后多算的点；全局网格间距不变。

缓存均在每个 model-period 开始时清空，额外内存计入 `workspace_bytes`。没有使用 FP32、fast math、较松容差或减少验证密度。

## 最终验证

| 环境 | 完整测试 |
|---|---:|
| macOS arm64，Python 3.12，CPU | 132 passed，3 CUDA-only skipped |
| Linux x86_64，Python 3.10，CPU | 132 passed，3 CUDA-only skipped |
| 同一 Linux 主机，A800/sm80，CUDA 12.6 | 135 passed |

三个档位都通过七套 0.2.1 逐位基准。独立物理参考检查使用全部 11 套归档：171 模型、6083 个周期轨迹、24332 个 mode cells，未排除任何 reference-unstable cell。CPU/CUDA、每个档位均为 **0 wrong / missing / extra / unresolved**，最大共同绝对误差均为 **2.586418306194105e-7 km/s**。

单独的极端 fixture 集仍保留四个 high-schedule unresolved 用例：三个 B10000 用例及 witness_model9384_period11。完整参考套件的零 unresolved 不代表这四个用例已恢复物理根。

CPU/CUDA 两个 sdist 均已实际重建 wheel，并各自通过 21 项逐位基准。

最终 CUDA 构建执行 memcheck、racecheck、synccheck、initcheck，各覆盖五个分块/缓存生命周期/块边界测试：全部 0 errors，racecheck 同时为 0 warnings。未把早期未实际进入 CUDA 的 sanitizer 尝试当成有效检查。

## 最终性能

同机交错执行，单独记录首次调用及 warmup，取五次 steady 中位数。每次输出都与同构建 high 的速度、NaN 位型、mask、status、flags、level、root_count、bounds **逐位相同**。

| 平台与工作量 | high | balanced | fast |
|---|---:|---:|---:|
| Linux，32 CPU 线程，128 模型 × 70 层 × 100 周期，5 模态 | 8.5940 s | **6.5761 s（1.307×）** | 6.5899 s（1.304×） |
| A800，256 模型 × 70 层 × 100 周期，5 模态 | 2.0316 s | 2.0325 s（1.000×） | 2.0344 s（0.999×） |

输入来自既有 tuning1024 的不重复前缀。上述最终 Linux 测量使用修复后完整源码。macOS 的早期 128 模型实验测得 balanced/fast 约 1.27×/1.35×；最终 wheel 的较小 smoke 也通过逐位等价检查，详见 receipt。结果不外推到完整 Base/Aug、其他 GPU 或其他层数。

## 未采用的 CUDA 方案

| 实验 | balanced 相对 high | fast 相对 high | 结论 |
|---|---:|---:|---|
| dense tile 改为 256/128，原 256 点缓存 | 1.007× | 0.965× | 无实质收益；fast 退化 |
| 再将精确采样缓存扩大至 2048 点 | 1.012× | 0.967× | 求值减少约 13%，但设备工作区由约 124 MB 增至 858 MB，不保留 |
| 再用缓存作大点集归并暂存区 | 1.011× | 0.964× | 实际执行 15201 次大归并，仍无实质收益，不保留 |

这些候选均通过独立参考与逐位性能结果检查，但通过正确性不等于值得采用。最终 CUDA 恢复 256 点缓存及 2048 点 dense tile。

后续值得单独验证的方向是融合两个 dense 网格的函数求值、减少重复排序及 kernel 调度；本次尚未实现或计入加速数字。不能再用减少验证代替这些优化。

## 复现和产物

- 数值 receipt：[accuracy-fix-20261028.json](benchmarks/accuracy-fix-20261028.json)。包含输入/扩展/源码哈希、每套参考指标、原始 timing samples、计数器和未采用实验。
- 运行完整测试：`python -m pytest tests -q`，CUDA 添加 `MODISP_TEST_BACKEND=cuda`。
- 性能驱动：`tools/benchmark_verified_presets.py`，用 `--references` 指定原有 11 套归档，`--input` / `--periods` 指定实际输入，`--backend` / `--threads` / `--models` 控制规模。输出目录必须不存在。任何错根、漏根、多根、unresolved 或与 high 不同的完整诊断都会使命令失败。
- 本机验证包、sdist、完整日志和服务器报告保存在 `/Users/yusong/workspace/Test/modisp-fix-results-20261028/`。服务器原地输出在 `/home/yuanys/workspace/modisp-fix-20261028/`。本次只上传获授权的修改源码；参考输入读取服务器已有副本。
- 测试 wheel 仍使用 0.3.0 版本号，保存在独立目录，未替换正式发布产物。Linux CUDA 测试 wheel 仅编译 sm80；通用源码仍保留原架构列表。安装测试产物时成对使用同次构建的 CPU/CUDA 包。

与 high 的逐位等价及这些参考检查支持已测范围内的正确性，不构成对任意层状模型求根完备性的数学证明。
