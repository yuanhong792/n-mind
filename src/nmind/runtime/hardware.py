"""硬件探测与设备降级。

规则 6：无 CUDA 时自动走 CPU，且不因为"没有 GPU"而拒绝运行。
这里只报告事实（核数、可用内存、torch 编译时是否带 CUDA、CPU 指令集），
不做任何"因此性能如何"的推断——性能结论一律来自实测。
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

try:                                  # psutil 是软依赖：缺了也能跑，只是少 RAM 数字
    import psutil
except ImportError:                   # pragma: no cover
    psutil = None                     # type: ignore[assignment]


@dataclass
class HardwareReport:
    platform: str
    machine: str
    python: str
    torch: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    cpu_features: dict[str, bool]
    ram_total_mb: float | None
    ram_available_mb: float | None
    cuda_available: bool
    torch_threads: int = 0
    cuda_devices: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def cpu_features() -> dict[str, bool]:
    """查询 torch 运行时实际启用的 CPU 指令集。

    torch.cpu 下的这些函数是下划线开头（各版本都在，但不是公开 API），
    因此全部 hasattr 兜底：查不到就报 False，绝不猜。
    记录它的原因很实在——本机是 Raptor Lake，AVX-512 被微码禁用，
    同一份代码在支持 AVX-512/AMX 的机器上 CPU 吞吐会明显不同，必须能区分开。
    """
    out = {"avx2": False, "avx512": False, "amx_tile": False, "vnni": False}
    probes = {"avx2": "_is_avx2_supported", "avx512": "_is_avx512_supported",
              "amx_tile": "_is_amx_tile_supported", "vnni": "_is_vnni_supported"}
    for key, fn in probes.items():
        try:
            if hasattr(torch.cpu, fn):
                out[key] = bool(getattr(torch.cpu, fn)())
        except Exception:             # noqa: BLE001 - 探测失败不影响主流程
            out[key] = False
    return out


def detect(device_pref: str = "auto", *, torch_threads: int = 0) -> tuple[torch.device, HardwareReport]:
    """返回 (设备, 报告)。报告里同时记回最终生效的线程数，便于复现说明。"""
    report = HardwareReport(
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        python=platform.python_version(),
        torch=torch.__version__,
        cpu_count_logical=os.cpu_count() or 1,
        cpu_count_physical=psutil.cpu_count(logical=False) if psutil else None,
        cpu_features=cpu_features(),
        ram_total_mb=None,
        ram_available_mb=None,
        cuda_available=False,
    )
    if psutil:
        vm = psutil.virtual_memory()
        report.ram_total_mb = round(vm.total / 2**20, 1)
        report.ram_available_mb = round(vm.available / 2**20, 1)

    if torch.cuda.is_available():
        report.cuda_available = True
        report.cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    if device_pref == "cuda":
        if not report.cuda_available:
            raise RuntimeError("配置要求 cuda，但本机 torch 不可用 CUDA；请改用 auto 或 cpu")
        device = torch.device("cuda")
    elif device_pref == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if report.cuda_available else "cpu")

    if device.type == "cpu":
        report.notes.append(
            "走 CPU 路径：" + ("按配置指定" if device_pref == "cpu" else "本机无可用 CUDA（受支持的默认，不是故障）"))
    if report.ram_available_mb is not None and report.ram_available_mb < 1024:
        report.notes.append("可用内存低于 1 GB：建议缩小 n_blocks / traj_len 或关掉持久记忆")
    if not report.cpu_features.get("avx512"):
        report.notes.append("未启用 AVX-512：CPU 吞吐上限低于支持该指令集的机器，跨机对比需注明")

    n_threads = int(torch_threads) if torch_threads else max(1, min(8, report.cpu_count_logical // 2))
    torch.set_num_threads(n_threads)
    report.torch_threads = int(torch.get_num_threads())
    if os.environ.get("OMP_NUM_THREADS") is None:
        os.environ["OMP_NUM_THREADS"] = str(report.torch_threads)
    report.notes.append(f"torch 线程数 = {report.torch_threads}（OMP_NUM_THREADS 已同步；仅供复现说明）")
    return device, report


def free_disk_mb(path: str = ".") -> float | None:
    try:
        return round(shutil.disk_usage(path).free / 2**20, 1)
    except OSError:
        return None
