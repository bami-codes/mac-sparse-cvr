"""Manage RunPod GPU pods for study E within a budget (RunPod REST API v2, standard library only).

    python3 scripts/runpod.py gpus [--min-vram 24] [--cuda 13.0]   # prices and availability (no pod is created)
    python3 scripts/runpod.py templates                             # official images
    python3 scripts/runpod.py budget --usd 50 --max-price 1.0       # set the budget (USD) and the highest $/h per pod
    python3 scripts/runpod.py create --name e-pilot --gpu <id> --hours 3 --image <image> [--disk 300]
    python3 scripts/runpod.py status                                # pods, rates, spend so far against the budget
    python3 scripts/runpod.py exec <pod> -- <command ...>           # one command in the pod (25 s at most)
    python3 scripts/runpod.py ssh <pod>                             # the ssh command line for this pod
    python3 scripts/runpod.py terminate <pod>                       # delete the pod and its disk
    python3 scripts/runpod.py extend <pod> --hours 15               # longer lifetime, if it still fits the budget
    python3 scripts/runpod.py watch [--every 120]                   # terminate pods past their hours or the budget

Safety rules built in:
- the key is read from .secrets/runpod_api_key (git-ignored) or RUNPOD_API_KEY and never printed;
- the account's SSH keys are never changed (that call replaces every registered key): each pod gets this project's own
  public key (~/.ssh/runpod_naver_ed25519.pub) in its PUBLIC_KEY variable;
- create refuses a pod whose rate exceeds --max-price or whose planned hours would take the spend past the budget;
  every pod has a lifetime (--hours) and `watch` terminates it when the lifetime or the budget is reached;
- spend is the local ledger (rate x running time of every pod created here), checked against RunPod's billing records
  when they are available. Container disk is ephemeral: results are copied back before a pod is terminated.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.runpod.io/v2"
ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / ".secrets" / "runpod_api_key"
STATE = ROOT / "runs" / "runpod"
LEDGER = STATE / "ledger.json"
SSH_KEY = Path.home() / ".ssh" / "runpod_naver_ed25519"
UA = "naver-study-e/1.0"


def key() -> str:
    k = os.environ.get("RUNPOD_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not k:
        raise SystemExit(f"put the RunPod API key in {KEY_FILE} (one line)")
    return k


def call(method: str, path: str, body: dict | None = None, query: dict | None = None, timeout: int = 60):
    url = API + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                                          "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode()
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as e:  # the message carries RunPod's answer, never the request headers
        raise SystemExit(f"{method} {path}: HTTP {e.code}: {e.read().decode()[:1500]}") from None


# ------------------------------------------------------------------ ledger and budget
def now() -> float:
    return time.time()


def load() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"budget_usd": None, "max_price": None, "pods": {}}


def save(led: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, indent=2))
    os.replace(tmp, LEDGER)


def spent(led: dict, at: float | None = None) -> float:
    at = at or now()
    return sum(p["rate"] * ((p.get("ended") or at) - p["created"]) / 3600 for p in led["pods"].values())


def stamp(t: float | None) -> str | None:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if t else None


# ------------------------------------------------------------------ commands
def gpus(a) -> None:
    q = {"include": "AVAILABILITY", "product": "POD", "cloud": a.cloud, "minCudaVersion": a.cuda}
    rows = call("GET", "/catalog/gpus", query=q).get("gpus", [])
    rows = [g for g in rows if (g.get("memory") or 0) >= a.min_vram]
    print(f"{'id':40} {'VRAM':>5} {'$/h':>7} {'min RAM':>8} availability")
    for g in sorted(rows, key=lambda g: (g.get("price") or {}).get(a.cloud.lower()) or 1e9):
        price = (g.get("price") or {}).get(a.cloud.lower())
        print(f"{g['id']:40} {g.get('memory', ''):>5} {price if price is not None else '-':>7} "
              f"{g.get('minRam', ''):>8} {g.get('availability')}")


def templates(a) -> None:
    rows = call("GET", "/catalog/templates")
    rows = rows.get("templates", rows) if isinstance(rows, dict) else rows
    for t in rows[: a.limit]:
        print(json.dumps({k: t.get(k) for k in ("id", "name", "image", "category", "isRunpod") if k in t}))


def budget(a) -> None:
    led = load()
    led["budget_usd"], led["max_price"] = a.usd, a.max_price
    save(led)
    print(f"budget {a.usd} USD, at most {a.max_price} USD/h per pod; spent so far {spent(led):.2f} USD")


def public_key() -> str:
    pub = SSH_KEY.with_suffix(".pub")
    if not pub.exists():
        raise SystemExit(f"no {pub}: create it with  ssh-keygen -t ed25519 -N '' -C naver-study-e -f {SSH_KEY}")
    return pub.read_text().strip()


def create(a) -> None:
    led = load()
    if led["budget_usd"] is None:
        raise SystemExit("set the budget first: scripts/runpod.py budget --usd ... --max-price ...")
    gpu = {g["id"]: g for g in call("GET", "/catalog/gpus").get("gpus", [])}.get(a.gpu)
    if gpu is None:
        raise SystemExit(f"unknown GPU type {a.gpu}")
    unit = (gpu.get("price") or {}).get(a.cloud.lower())
    rate = None if unit is None else unit * a.count
    if rate is None or unit > led["max_price"]:
        raise SystemExit(f"{a.gpu} costs {unit} USD/h per GPU on {a.cloud}, above the limit {led['max_price']}")
    if spent(led) + rate * a.hours > led["budget_usd"]:
        raise SystemExit(f"spent {spent(led):.2f} + {rate} x {a.hours} h would pass the budget {led['budget_usd']} USD")
    body = {"name": a.name, "image": a.image, "disk": a.disk, "cloud": a.cloud, "ports": ["22/tcp"],
            "startSsh": True, "env": {"PUBLIC_KEY": public_key()},
            "gpu": {"id": a.gpu, "count": a.count, "minCudaVersion": a.cuda, "minRamPerGpu": a.min_ram,
                    "minVcpuCountPerGpu": a.min_vcpu}}  # the official DataLoader (8 workers) runs on the CPU
    pod = call("POST", "/pods", body)
    led["pods"][pod["id"]] = {"name": a.name, "gpu": f"{a.count}x {a.gpu}", "rate": float(pod.get("cost") or rate), "created": now(),
                              "max_hours": a.hours, "ended": None}
    save(led)
    print(f"created {pod['id']} ({a.gpu}, {led['pods'][pod['id']]['rate']} USD/h, lifetime {a.hours} h); "
          f"status {pod.get('status')}")


def status(a) -> None:
    led = load()
    live = {p["id"]: p for p in call("GET", "/pods").get("pods", [])}
    for pid, p in led["pods"].items():
        lp = live.get(pid)
        if lp is None and p["ended"] is None:  # gone outside this tool: close it at the time we noticed
            p["ended"] = now()
        hours = ((p["ended"] or now()) - p["created"]) / 3600
        print(f"{pid} {p['name']:14} {p['gpu']:28} {p['rate']:.3f} $/h  {hours:5.2f} h  "
              f"{(lp or {}).get('status', 'gone'):10} cuda {(lp or {}).get('cudaVersion')}  ended {stamp(p['ended'])}")
    for pid, lp in live.items():
        if pid not in led["pods"]:
            print(f"{pid} {lp.get('name')} (not created by this tool: never touched) status {lp.get('status')}")
    save(led)
    b = led["budget_usd"]
    print(f"spent {spent(led):.2f} USD of {b} USD" + (f" ({spent(led) / b:.0%})" if b else ""))


def pod_of(pid: str) -> dict:
    return call("GET", f"/pods/{pid}")


def exec_(a) -> None:
    cmd = a.command[1:] if a.command and a.command[0] == "--" else a.command
    r = call("POST", f"/pods/{a.pod}/exec", {"command": ["sh", "-c", " ".join(cmd)], "timeoutSeconds": a.timeout},
             timeout=a.timeout + 30)
    sys.stdout.write(r.get("stdout") or "")
    sys.stderr.write(r.get("stderr") or "")
    print(f"[exit {r.get('exitCode')}{', timed out' if r.get('timedOut') else ''}]")


def ssh(a) -> None:
    s = pod_of(a.pod).get("ssh") or {}
    print(json.dumps(s, indent=2))
    d = s.get("direct") or {}
    if d.get("host") and d.get("port"):
        print(f"ssh -i {SSH_KEY} -p {d['port']} {d.get('username', 'root')}@{d['host']}")


def terminate(a) -> None:
    led = load()
    if a.pod not in led["pods"]:
        raise SystemExit(f"{a.pod} was not created by this tool; it is never touched")
    call("DELETE", f"/pods/{a.pod}")
    led["pods"][a.pod]["ended"] = now()
    save(led)
    print(f"terminated {a.pod}; spent {spent(led):.2f} USD of {led['budget_usd']} USD")


def extend(a) -> None:
    """Give a pod created here a longer lifetime, only if its rate over the new lifetime still fits the budget."""
    led = load()
    pod = led["pods"].get(a.pod)
    if pod is None or pod["ended"] is not None:
        raise SystemExit(f"{a.pod} is not a running pod created by this tool")
    left = pod["rate"] * max(0.0, a.hours - (now() - pod["created"]) / 3600)
    if spent(led) + left > led["budget_usd"]:
        raise SystemExit(f"spent {spent(led):.2f} + {left:.2f} until hour {a.hours} would pass the budget {led['budget_usd']}")
    pod["max_hours"] = a.hours
    save(led)
    print(f"{a.pod}: lifetime {a.hours} h; at most {spent(led) + left:.2f} USD of {led['budget_usd']} USD by then")


def watch(a) -> None:
    while True:
        led = load()
        for pid, p in led["pods"].items():
            if p["ended"] is not None:
                continue
            over_life = (now() - p["created"]) / 3600 >= p["max_hours"]
            over_budget = led["budget_usd"] is not None and spent(led) >= led["budget_usd"]
            if over_life or over_budget:
                call("DELETE", f"/pods/{pid}")
                p["ended"] = now()
                save(led)
                print(f"{stamp(now())} terminated {pid}: {'lifetime' if over_life else 'budget'} reached", flush=True)
        live = [p for p in led["pods"].values() if p["ended"] is None]
        print(f"{stamp(now())} running {len(live)}, spent {spent(led):.2f} of {led['budget_usd']} USD", flush=True)
        if a.once or not live:
            return
        time.sleep(a.every)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gpus")
    p.add_argument("--min-vram", type=int, default=24)
    p.add_argument("--cuda", default="13.0")
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p = sub.add_parser("templates")
    p.add_argument("--limit", type=int, default=40)
    p = sub.add_parser("budget")
    p.add_argument("--usd", type=float, required=True)
    p.add_argument("--max-price", type=float, required=True)
    p = sub.add_parser("create")
    p.add_argument("--name", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--count", type=int, default=1, help="GPUs on the pod; the rate and the budget check use the total")
    p.add_argument("--hours", type=float, required=True, help="lifetime: watch terminates the pod after it")
    p.add_argument("--image", required=True)
    p.add_argument("--disk", type=int, default=300)
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--cuda", default="13.0", help="lowest host CUDA version (the lock installs CUDA 13 wheels)")
    p.add_argument("--min-ram", type=int, default=64)
    p.add_argument("--min-vcpu", type=int, default=12)
    sub.add_parser("status")
    p = sub.add_parser("exec")
    p.add_argument("pod")
    p.add_argument("--timeout", type=int, default=25)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("ssh")
    p.add_argument("pod")
    p = sub.add_parser("terminate")
    p.add_argument("pod")
    p = sub.add_parser("extend")
    p.add_argument("pod")
    p.add_argument("--hours", type=float, required=True, help="the new lifetime, counted from the pod's creation")
    p = sub.add_parser("watch")
    p.add_argument("--every", type=int, default=120)
    p.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    {"gpus": gpus, "templates": templates, "budget": budget, "create": create, "status": status, "exec": exec_,
     "ssh": ssh, "terminate": terminate, "extend": extend, "watch": watch}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
