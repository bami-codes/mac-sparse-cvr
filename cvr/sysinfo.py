"""Machine facts recorded with each run, the same way on macOS and Linux."""

import platform
import resource
import subprocess
import sys
from pathlib import Path


def peak_rss_gb() -> float:
    """Peak resident memory of this process in GB. getrusage reports bytes on macOS but kilobytes on Linux."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((r if sys.platform == "darwin" else r * 1024) / 1e9, 2)


def cpu_name() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip()
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def container_peak_memory_gb() -> float | None:
    """Peak memory of the whole container (every process, e.g. the 8 DataLoader workers) from the Linux cgroup,
    or None where there is no cgroup memory file (macOS). Covers the time since the container started."""
    for f, scale in (("/sys/fs/cgroup/memory.peak", 1), ("/sys/fs/cgroup/memory/memory.max_usage_in_bytes", 1)):
        p = Path(f)
        if p.exists():
            try:
                return round(int(p.read_text().split()[0]) * scale / 1e9, 2)
            except (OSError, ValueError):
                return None
    return None
