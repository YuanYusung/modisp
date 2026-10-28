# balanced/fast 观测尺度加速验证（2026-10-28）

已按本次选定的相对速度差 **0.5%** 实现可配置加速。`high` 保留原 0.2.1 的完整搜索及验证路径；`balanced` 和 `fast` 使用与观测尺度关联的速度网格。它们允许近根簇合并、严格根值差异和后续序号偏移，并在结果中保留不确定性标记。

本实现发布为 **v0.3.0**，私有 ABI 为 **4**；CPU 与 CUDA 插件需要使用本次配套构建。早期本地 0.3.0 候选包使用 ABI 3，需同时替换。先前严格修复实验作为历史记录保留。

## 改动与两项问题的处理

原版 0.3.0 的两个问题是：稀疏搜索提前接受可能跳过根；粗网格结果一致可能被当作已完整验证。先前严格修复实验要求所有档位执行完整验证，保存于 `CORRECTNESS_PERFORMANCE_FIX.md`。本次按用户允许的近根容忍范围重新设计 balanced/fast，取消基于“已找到根间距”的完整性判断，改为直接限制采样速度间隔，并明确标记所有近似搜索。

曾尝试恢复较宽松的提前接受，但随机 Love 模型中发现间隔约 1.65% 的根仍会漏失，因此未采用。该反例已加入独立原始 SH 2×2 递推加 Brent 求根的回归测试；目前三个档位均通过。

| 档位 | 搜索 | 默认最大相对速度步长 | 默认精化容差 `c_tol_rel` |
|---|---|---:|---:|
| high | 原完整 adaptive/guard/dense 路径 | 原策略 | 1e-6 |
| balanced | 对数速度网格＋临界速度/界面根探针 | 0.25% | 1e-5 |
| fast | 较粗对数速度网格＋相同探针 | 0.5% | 1e-4 |

步长分别为 `0.5*close_root_rtol` 和 `close_root_rtol`；默认 `close_root_rtol=0.005`。函数计算和根精化仍为 FP64，性能提升主要来自减少重复搜索与函数求值，并适当放宽精化精度。CPU 分块扫描，CUDA 使用融合采样、括根和精化的持久 block。搜索内部多收集两个 guard roots，返回请求数量的已恢复根。显式 `c_tol_rel` 仍可覆盖档位默认值。

临界速度探针不会扩大网格间隔；Love 忽略无黏水层的传播速度探针，保持水深无关性。为完全避免原版“安全接受”语义，`early_acceptance_enabled=False`。旧参数 `adaptive_period_threshold` 保留为兼容元数据，不再决定新网格。

## 性能

yaolab-server，Intel Xeon Gold 6430，CPU 使用 32 个求解线程；GPU 为 NVIDIA A800 80GB PCIe、CUDA 12.6/sm_80。每组分别测量首调用、预热，再做 **5 次交错顺序的稳态重复**，下表为中位数；每次输出均执行相同质量检查。

| 后端与工作量 | high | balanced | 相对 high | fast | 相对 high |
|---|---:|---:|---:|---:|---:|
| CPU：128 个不同模型×70 层×100 周期×5 模态 | 8.2963 s | 2.6343 s | **3.15×** | 1.5344 s | **5.41×** |
| CUDA：1024 个不同模型×70 层×100 周期×5 模态 | 7.4915 s | 2.0623 s | **3.63×** | 1.5227 s | **4.92×** |

两行批量不同，应分别比较同一行的档位，不能据此直接计算 CPU/GPU 加速比。CPU 的 128 个模型为 1024 输入集的前缀；JSON 保留完整输入文件及周期文件 SHA256、每次计时、首调用时间、求值计数和扩展二进制哈希。这些数字仅适用于上述工作量与环境。

## 如何判断结果

`0.5%` 是本次选择的观测尺度参数，不是每个带编号模态速度的最大误差保证，也不是特定实际观测系统的分辨率实测值。近根漏失后，后续返回序号可能对应另一个物理模态，即使后续根本身数值准确。若逐阶模态标签必须稳定，使用 `high`。

```python
from modisp import Solver

with Solver(backend="auto", accuracy="balanced", close_root_rtol=0.005) as solver:
    result = solver(thickness, vp, vs, rho, periods, nmodes=5)
    velocity = result.phase_velocity
    approximate = result.approximate_search       # [B,T]
    uncertain_rank = result.mode_index_uncertain  # [B,T]
    close_hint = result.close_root_risk            # [B,T]
    observed_pairs = result.close_root_mask        # [B,M,T]
```

- `TRACE_APPROXIMATE_SEARCH=2048`：所有实际执行观测尺度网格的 trace 都置位，包括未解成功的尝试。解析空集/空窗口可以保持精确空状态。
- `mode_index_uncertain` 等同于近似搜索标记，是保守的序号不确定性提示；并非每条 trace 都出错。
- `TRACE_CLOSE_ROOT_RISK=4096` / `close_root_risk`：已恢复的近根或采样低绝对值同号谷值提示；CPU/CUDA 分块不同可能产生不同的提示数量。
- `close_root_mask` 只标出已恢复结果中相邻间距在阈值内的根。`close_root_risk=False` 或 mask 为 False **不能排除未采到的近根对**。保守的整体近似标记始终保留。
- `potentially_ambiguous` 合并近似与近根风险标记。`VALID` 仅表示当前策略接受有限值；近似 `NOT_FOUND` 不能证明物理模态不存在。

CLI 提供 `--close-root-rtol` 和 `--c-tol-rel`，输出 NPZ 保存这些质量数组及阈值。需要严格结果时可用同一模型/周期以 `accuracy="high"` 重算。

## 验证判据及证据边界

按参考根的相邻相对间距形成可传递近根簇；簇外孤立根必须在结果集合中出现，balanced/fast 相对误差预算分别为 `2e-5`/`2e-4`。与普通参考根和允许的近根簇包络都不匹配的结果计为伪根，**即使已有风险标记也会失败**。匹配使用根集合，不会把近根漏失后的所有序号差异都当作普通速度误差。

统计同时保留逐序号差异、缺项、近根簇严格未匹配、簇近似、序号偏移和未判定输出。`close_cluster_missing` 表示该近根簇没有任何成员达到严格细根匹配容差；仍可能存在符合观测尺度的簇代表值，因此不能直接解释为整个可观测簇消失。`max_matched_relative_error` 仅为已严格匹配根的最大误差，不覆盖簇近似及未判定值。

### 11 组独立固体 Rayleigh 参考

共 171 模型、6083 条 model-period trace、24332 个主参考单元。原独立参考前四阶保持不变；只额外拼接同物理 `high` 的第 5–8 根，用于判定跨参考前缀边界的近根簇。这些 guard roots **不属于新增独立物理证据**。CPU 和 CUDA 的全部 6083 条 guard 前缀均与原参考一致，无前缀不一致。

| 档位（CPU/CUDA 相同，局部风险提示数可不同） | 孤立根漏失＋普通伪根 | 逐序号差异 | 逐序号缺项 | 近根簇严格未匹配 | 序号偏移 | 超出已知参考前缀的未判定值 |
|---|---:|---:|---:|---:|---:|---:|
| high | 0 | 0 | 0 | 0 | 0 | 0 |
| balanced | 0 | 40 | 0 | 11 | 28 | 7 |
| fast | 0 | 93 | 0 | 25 | 73 | 17 |

所有近似 trace 均有标记；无未标记不确定性，也无新增 UNRESOLVED。已严格匹配根最大相对差约为 balanced `6.27e-7`、fast `6.28e-6`。超出前缀的 7/17 个值保留为未判定，未宣称正确或错误。

### 性能工作量上的质量

| 后端/档位 | 孤立根漏失＋普通伪根 | 逐序号差异 | 逐序号缺项 | 近根簇严格未匹配 | 序号偏移 | 超参考前缀未判定值 |
|---|---:|---:|---:|---:|---:|---:|
| CPU balanced | 0 | 450 | 0 | 2 | 381 | 24 |
| CPU fast | 0 | 1032 | 0 | 63 | 821 | 177 |
| CUDA balanced | 0 | 2009 | 0 | 13 | 1660 | 123 |
| CUDA fast | 0 | 4942 | 2 | 314 | 3927 | 860 |

CPU 共 12800 trace，无 high 基线未解；CUDA 共 102400 trace，其中 high 基线有 1 条未解，剩余 **102399 条**可按参考判定。该条上的每档 5 个有限近似值单列为未判定。CUDA fast 另有 1 个参考末端无足够右侧 guard 的边界情况，保留 `unjudged_reference_boundary=1`。所有不确定情况都有标记，近似档位没有新增 UNRESOLVED。已严格匹配根最大相对差 CPU 为 `6.20e-7`/`6.25e-6`，CUDA 为 `6.74e-7`/`6.28e-6`，均不代表全体逐序号输出的误差上界。

### 随机 Rayleigh/Love 压力回归

种子 3490–3493，各 160 个六层模型、24 个 0.3–30 s 周期，同时计算 Rayleigh/Love。每后端共 **30720 条 trace**，用 high8 对照近似5；这是同物理高精度回归，不是独立物理真值。

high 有 117 条未解，因此仅 **30603 条**进入根判定；每档在未解行上的 585 个有限值单列为未判定，不做正确性结论。

| 后端/档位 | 孤立根漏失＋普通伪根 | 逐序号差异 | 逐序号缺项 | 近根簇严格未匹配 | 序号偏移 | 超参考前缀未判定值 |
|---|---:|---:|---:|---:|---:|---:|
| CPU balanced | 0 | 11599 | 2 | 220 | 9156 | 1283 |
| CPU fast | 0 | 19418 | 6 | 1002 | 13371 | 5288 |
| CUDA balanced | 0 | 11583 | 2 | 218 | 9140 | 1283 |
| CUDA fast | 0 | 19408 | 6 | 999 | 13358 | 5290 |

这一压力集故意包含许多近根簇，序号差异明显大于性能集，说明不能把“普通根判据通过”等同于所有编号模态正确。可判定范围内未发现超过阈值的孤立根漏失/普通伪根；所有不确定性有标记。此结果不是任意模型、任意参数的完备性证明。

## 测试、同步与打包

- macOS CPU：140 passed，3 个 CUDA 专项 skipped；Linux CPU：140 passed，3 skipped。
- Linux CUDA：143 passed。包括原 high 七组 0.2.1 位级回归、两类原始失败反例、独立孤立 Love 根、解析/第三方简单模型、分块/复用、阈值、CLI、128/256 launch、256 层边界。
- 最终 CUDA 代码分别执行 Compute Sanitizer memcheck、racecheck、synccheck、initcheck；每项覆盖 6 个针对性测试，全部通过，0 errors，racecheck 0 warnings。
- 验证中修复了新融合 kernel 的共享尾部未清零，以及读取共享采样数量后缺少 block 屏障的问题。上述 CUDA 全量测试、压力回归及性能数字均来自修复后的构建。
- CUDA 插件使用规范共享源码；发行准备脚本同步自包含 `cpp`，避免旧副本漂移。ABI 4 阻止旧插件混用。发布附件中的 Linux CUDA wheel 针对 sm_80，其他支持架构可由源码构建。

完整 JSON 见 [tolerant-presets-20261028.json](benchmarks/tolerant-presets-20261028.json)，保留原始四份报告、聚合数据、运行环境和哈希。复现入口为 `tools/benchmark_tolerant_presets.py`、`tools/validate_tolerant_stress.py`，判据实现为 `tools/tolerant_metrics.py`。服务器只上传了获授权源码，独立参考和性能输入使用服务器原有副本。

源码、配套 wheel、sdist、报告、数值输出、检查日志和 SHA256 清单见 [GitHub v0.3.0 发布附件](https://github.com/YuanYusung/modisp/releases/tag/v0.3.0)。本地与服务器原始验证记录及先前严格修复实验继续保留。
