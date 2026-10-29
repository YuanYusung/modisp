# 中文快速开始

modisp 提供 CPU 基础包和可选 CUDA 12 插件，支持固体 Rayleigh、表面水载
P–SV/Scholte 型根及 Love 多模态相速度。

## 安装

```sh
python -m pip install /path/to/modisp-0.3.2-<匹配标签>.whl
modisp doctor
```

CPU 使用不需要 CUDA。有 NVIDIA GPU 时，再安装版本完全一致的
`modisp_cuda12` wheel。离线安装：

```sh
python -m pip install --no-index --find-links /path/to/wheelhouse modisp==0.3.2
```

从源码安装使用 `python -m pip install <CPU源码包.tar.gz>`，需要 C++17
编译器和 CMake；CUDA 源码构建另需 CUDA 12 工具链。详见[安装说明](INSTALL.md)。

## 统一的单模型与批量接口

`dispersion`、`dispersion_batch` 和 `Solver` 都返回完整的
`DispersionBatch`，单模型只是 `B=1` 的批量。

```python
import numpy as np
from modisp import dispersion

# 厚度 km、Vp/Vs km/s、密度 g/cm³；最后一行是半空间。
model = np.array([[0.3, 2.0, 0.8, 2.0], [0.0, 5.0, 2.8, 2.6]])
periods = np.logspace(-1, 1, 30)  # 秒，允许乱序和重复
result = dispersion(model, periods, nmodes=2, wave="rayleigh",
                    backend="cpu", threads=4)
rayleigh = result.phase_velocity[0, 0]
print(result.period, rayleigh)
print(result.status, result.trace_flags)
```

未检出的点保留 NaN 和 `NOT_FOUND`；数值未收敛或接受失败保留 NaN 和
`UNRESOLVED`/`REJECTED`，不再过滤周期或抛出单曲线专用异常。

从 0.1.0 迁移时，将 `pd(periods, mode=m)` 改为
`dispersion(model, periods, nmodes=m+1)`，将 `curve.velocity` 改为
`result.phase_velocity[0, m]`。

## 水层和混合模型

```python
water_model = np.vstack([[0.025, 1.5, 0.0, 1.02], model])
result = dispersion(water_model, periods, nmodes=3, backend="auto")
print(result.phase_velocity.shape)  # [1, 3, 30]
print(result.backend_used, result.status, result.trace_flags)
```

首层 `Vs=0` 表示声学流体，使用给定的水深、Vp 和密度。含水时 `rayleigh`
表示耦合 P–SV 根族，包括 Scholte 型低速根。无黏水层不改变 Love 相速度。
适用范围与不包含的功能见 [README](../README.md#scope-and-numerical-diagnostics)。

## 复用 Solver

```python
from modisp import Solver

with Solver(backend="auto", threads=8, chunk_size=100) as solver:
    batch = solver(thickness, vp, vs, rho, periods, nmodes=3, wave="rayleigh")
    print(batch.phase_velocity.shape)  # [模型数, 模态数, 周期数]
    print(solver.stats(), solver.last_result is batch)
```

模型参数为相同形状的 `[B,L]` 数组；层数不同可用 `to_batch` 插入零厚度
固体层。`auto` 在匹配插件及设备可用时用 CUDA，否则用 CPU 并记录原因。
显式 `cuda` 不可用时报错，数值失败不会切换后端。同一 Solver 不可并发调用；
`close()` 或 `with` 释放资源。

## 精度档位

默认 `balanced` 和 `fast` 使用局部自适应采样，分别允许在不一致区间宽度达到近根尺度的 0.5 倍和 1 倍时停止细分；
二者仍使用 FP64，但允许不可分辨近根簇发生遗漏或近似，换取更高速度。

```python
with Solver(backend="auto", accuracy="balanced", close_root_rtol=0.005) as solver:
    result = solver(thickness, vp, vs, rho, periods, nmodes=5)
    approximate = result.approximate_search    # 所有近似搜索结果均明确标记
    uncertain_rank = result.mode_index_uncertain
    close = result.close_root_mask             # 已恢复根中的相邻近根
```

`close_root_rtol=0.005` 表示 0.5%，可配置。`high` 逐根搜索，不按此阈值合并。
0.3.2 默认 `search_strategy="shape_adaptive"`，用局部五点积分一致性决定细分。
`secular=None` 按实际后端选择：CUDA 默认 `fast`，CPU 默认 `dunkin`；
`backend="auto"` 回退 CPU 时也采用 `dunkin`。legacy `uniform` 默认仍用 `dunkin`。
显式 `secular="fast"` 或 `"dunkin"` 可覆盖自适应求解器的默认选择。固定节点 CPU 测量约 1.03–1.19× 函数加速、1.07–1.10×
求解器加速，A800 1,024 模型约 1.33×，详见 [0.3.2 说明](SECULAR_FAST_0_3_2.md)。
`sampling_rtol` 三档默认 `1e-3`／`1e-2`／`5e-2`，与速度精化容差和近根尺度分别管理。
需要复现 0.3.0 时显式选择 `search_strategy="qe_adaptive"`。
实现、独立验证和性能测量见 [0.3.1 形状自适应说明](SHAPE_ADAPTIVE_0_3_1.md) 和 [0.3.2 快速频散函数说明](SECULAR_FAST_0_3_2.md)。
近似档的 `mode` 是恢复根的排序编号；遗漏近根簇后可能错位。`close_root_mask=False`
不能证明没有漏掉近根。默认根精修容差为 high `1e-6`、balanced `1e-5`、fast `1e-4`；
显式 `c_tol_rel` 仍优先。`search_level=8/9/10` 分别表示 high/balanced/fast 形状搜索；6/7 为旧路径的观测网格。
详见[当前精度与速度报告](SECULAR_FAST_0_3_2.md) 与 [0.3.1 验证](SHAPE_ADAPTIVE_0_3_1.md)。

## 诊断和编号

`status`：0 有效，1 完成搜索但未检出，2 数值未收敛，3 接受条件拒绝。
`mode_mask` 标记有限输出，`trace_flags` 是按位累计诊断：

| 位值 | 含义 |
|---:|---|
| 4 | 最终搜索未收敛，整条周期轨迹输出 NaN |
| 8 | 搜索采样或工作量达到上限；后续回退成功时可同时存在有效输出 |
| 16 | 请求根接近半空间 Vs 截止 |
| 32 / 64 | 非有限或数值塌缩 / 精修失败 |
| 128 | 接受条件拒绝 |
| 256 | 用户显式给定搜索边界 |
| 512 | 密度反解未通过重新求值的间距检查 |
| 1024 | 采样族结果出现分歧；结合最终状态判断 |
| 8192 | 形状搜索回退到对应的 0.3.0 路径 |

根按每个周期的相速度升序编号，不追踪跨周期物理分支。显式 `cmin/cmax`
会裁切搜索谱，此时模式 0 仅表示窗口内首根。`root_count` 至多计入请求模态
加 2 个保护根。`search_level=3` 表示自适应检查通过，5 表示密集回退通过，
-3 表示 Love 根族解析为空，-4 表示显式窗口与搜索域不相交。

## 命令行

```sh
python examples/make_input.py input.npz
modisp forward input.npz --output result.npz --backend cpu --threads 4 --nmodes 3
modisp doctor
```

NPZ 输出保留速度、mask、状态、标志、搜索范围和实际后端。未收敛时退出码
为 2，文件仍完整写出。完整参数说明见 [Python API](API.md)。
