"""The machine's NVIDIA GPUs: name, compute capability, memory and the processes using them.

``discover_gpus`` asks NVML (``pynvml``, from nvidia-ml-py) first, then ``nvidia-smi``'s CSV queries, then
``torch.cuda`` (no process list and no free memory, since measuring it would create a CUDA context on every GPU).

**Indices:** ``GPU.index`` is the index this process and its children see: with ``CUDA_VISIBLE_DEVICES`` set, the
listed GPUs in its order (indices or UUIDs), numbered from 0; without it, NVML's order. ``GPU.physical_index`` is
NVML's and ``nvidia-smi``'s. ``visible_only=False`` lists every GPU with ``index == physical_index``.
CUDA orders devices "fastest first" unless ``CUDA_DEVICE_ORDER=PCI_BUS_ID``; on a machine of identical GPUs the two
orders agree.

Memory is in bytes. In a container NVML may list no processes (they are in another PID namespace); their memory
still counts as used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace

MIB = 1024**2


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    used: int | None  # bytes; None when the driver does not say


@dataclass(frozen=True)
class GPU:
    index: int  # as this process sees it (CUDA_VISIBLE_DEVICES)
    physical_index: int  # NVML's / nvidia-smi's
    uuid: str
    name: str
    compute_capability: str | None  # "9.0"
    memory_total: int  # bytes
    memory_used: int | None
    memory_free: int | None
    utilization: int | None = None  # percent
    processes: tuple[GPUProcess, ...] = field(default_factory=tuple)
    source: str = "nvml"  # nvml, nvidia-smi or torch

    def describe(self) -> dict:
        return {**asdict(self), "processes": [asdict(p) for p in self.processes]}


def discover_gpus(visible_only: bool = True) -> list[GPU]:
    """The GPUs, from the first probe that answers (NVML, ``nvidia-smi``, ``torch.cuda``); [] without any."""
    for probe in (_nvml_gpus, _smi_gpus):
        try:
            gpus = probe()
        except Exception:
            continue
        return visible(gpus, os.environ.get("CUDA_VISIBLE_DEVICES")) if visible_only else gpus
    try:
        return _torch_gpus()  # already numbered as CUDA sees them
    except Exception:
        return []


def physical_gpu(physical_index: int) -> GPU:
    """One GPU by its NVML / ``nvidia-smi`` index, whatever ``CUDA_VISIBLE_DEVICES`` says (``LookupError``)."""
    for gpu in discover_gpus(visible_only=False):
        if gpu.physical_index == int(physical_index):
            return gpu
    raise LookupError(f"no GPU {physical_index}")


def visible(gpus: list[GPU], cuda_visible_devices: str | None) -> list[GPU]:
    """``gpus`` (physical order) as CUDA sees them under ``cuda_visible_devices``: the listed ones, renumbered from 0.

    Entries are indices or UUIDs (a unique prefix such as ``GPU-8932f937`` is enough); CUDA stops at the first
    entry it cannot resolve, and so does this. Unset: every GPU, numbered as NVML numbers them."""
    if cuda_visible_devices is None:
        return [replace(g, index=g.physical_index) for g in gpus]
    chosen: list[GPU] = []
    for entry in (e.strip() for e in cuda_visible_devices.split(",")):
        match = None
        if entry.lstrip("-").isdigit():
            match = next((g for g in gpus if g.physical_index == int(entry)), None)
        elif entry:
            found = [g for g in gpus if g.uuid.startswith(entry) or g.uuid.removeprefix("GPU-").startswith(entry)]
            match = found[0] if len(found) == 1 else None
        if match is None or match in chosen:
            break
        chosen.append(match)
    return [replace(g, index=i) for i, g in enumerate(chosen)]


# ---------------------------------------------------------------------------------------------- probes


def _nvml_gpus() -> list[GPU]:
    import pynvml

    pynvml.nvmlInit()
    try:
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                capability: str | None = f"{major}.{minor}"
            except pynvml.NVMLError:
                capability = None
            try:
                utilization: int | None = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            except pynvml.NVMLError:
                utilization = None
            try:
                running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError:
                running = []
            processes = tuple(GPUProcess(int(p.pid), int(p.usedGpuMemory) if p.usedGpuMemory is not None else None)
                              for p in running)
            gpus.append(GPU(index=i, physical_index=i, uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
                            name=_text(pynvml.nvmlDeviceGetName(handle)), compute_capability=capability,
                            memory_total=int(memory.total), memory_used=int(memory.used),
                            memory_free=int(memory.free), utilization=utilization, processes=processes))
        return gpus
    finally:
        pynvml.nvmlShutdown()


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


SMI_GPU_FIELDS = "index,uuid,name,compute_cap,memory.total,memory.used,memory.free,utilization.gpu"
SMI_APP_FIELDS = "gpu_uuid,pid,used_memory"


def _smi_gpus() -> list[GPU]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise FileNotFoundError("nvidia-smi")

    def query(kind, fields):
        return subprocess.run([smi, f"--query-{kind}={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                              text=True, check=True, timeout=20).stdout

    try:
        gpus = query("gpu", SMI_GPU_FIELDS)
    except subprocess.CalledProcessError:  # a driver without compute_cap
        gpus = query("gpu", SMI_GPU_FIELDS.replace("compute_cap,", ""))
    try:
        apps = query("compute-apps", SMI_APP_FIELDS)
    except subprocess.CalledProcessError:
        apps = ""
    return parse_nvidia_smi(gpus, apps)


def parse_nvidia_smi(gpus_csv: str, apps_csv: str = "") -> list[GPU]:
    """GPUs from ``nvidia-smi --query-gpu=<SMI_GPU_FIELDS>`` and ``--query-compute-apps=<SMI_APP_FIELDS>``
    (``--format=csv,noheader,nounits``: MiB, percent). A query without ``compute_cap`` (7 columns) is accepted."""

    def number(value):
        value = value.strip()
        try:
            return int(float(value))
        except ValueError:  # [N/A], [Not Supported]
            return None

    processes: dict[str, list[GPUProcess]] = {}
    for line in apps_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and number(parts[1]) is not None:
            used = number(parts[2])
            processes.setdefault(parts[0], []).append(GPUProcess(number(parts[1]), used * MIB if used is not None
                                                                 else None))
    result = []
    for line in gpus_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7:
            parts.insert(3, "")
        if len(parts) < 8 or number(parts[0]) is None:
            continue
        index, uuid, name, capability, total, used, free, utilization = parts[:8]
        mib = [number(v) for v in (total, used, free)]
        result.append(GPU(index=number(index), physical_index=number(index), uuid=uuid, name=name,
                          compute_capability=capability or None, memory_total=(mib[0] or 0) * MIB,
                          memory_used=mib[1] * MIB if mib[1] is not None else None,
                          memory_free=mib[2] * MIB if mib[2] is not None else None,
                          utilization=number(utilization), processes=tuple(processes.get(uuid, ())),
                          source="nvidia-smi"))
    return result


def _torch_gpus() -> list[GPU]:
    import torch

    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        uuid = str(getattr(p, "uuid", "") or "")
        gpus.append(GPU(index=i, physical_index=i, uuid=uuid if uuid.startswith("GPU-") or not uuid else f"GPU-{uuid}",
                        name=p.name, compute_capability=f"{p.major}.{p.minor}", memory_total=int(p.total_memory),
                        memory_used=None, memory_free=None, source="torch"))
    return gpus


__all__ = ["GPU", "GPUProcess", "discover_gpus", "parse_nvidia_smi", "physical_gpu", "visible"]
