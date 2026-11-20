"""Print '<left> <total> <queue up|down>' for a job file of scripts/pod_queue.py, or 'missing 0 <queue>'.

    python scripts/pod_left.py runs/train/jobs_grid.json

'left' counts the jobs that are neither done nor diverged, using pod_queue's own state rules. The queue is looked
up with a pattern that cannot match this process or the shell that started it: a plain 'pod_queue.py run' matches
the ssh command line itself, and a stopped queue then looks alive.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main(path: str) -> str:
    up = subprocess.run(["pgrep", "-f", "pod_queue[.]py run"], capture_output=True).returncode == 0
    queue = "up" if up else "down"
    if not Path(path).exists():
        return f"missing 0 {queue}"
    import pod_queue as Q

    jobs = json.loads(Path(path).read_text())
    left = sum(Q.state(j) not in ("done", "diverged") for j in jobs)
    return f"{left} {len(jobs)} {queue}"


if __name__ == "__main__":
    print(main(sys.argv[1]))
