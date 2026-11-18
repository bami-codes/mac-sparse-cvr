"""Machine facts recorded with each run, the same way on macOS and Linux."""

import platform
import resource
import subprocess
import sys
from pathlib import Path


def peak_rss_gb() -> float:
    """Peak resident memory of this process in GB. getrusage reports bytes on macOS and kilobytes on Linux
    (the runs used to divide by 1e9 everywhere, which is 1000 times too small on Linux)."""
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
