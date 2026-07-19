#!/usr/bin/env python3
"""Vast.ai GPU-box lifecycle CLI — search / rent / wait-ssh / provision / link-pi / destroy.

Automates the manual loop we run for every ephemeral GPU box (see
docs/vast_coordination.md for the multi-agent protocol this tool enforces):

    vast_box.py search                          # ranked offers (4090 default)
    vast_box.py rent <offer_id> --label mert    # template 405071, ledger-tracked
    vast_box.py wait-ssh <instance_id>          # poll → probe 22 → rewrite ~/.ssh/config
    vast_box.py provision --id <instance_id>    # GitHub clone + vast_bootstrap.sh
    vast_box.py link-pi --id <instance_id>      # box key + tailscale (2 human steps)
    vast_box.py destroy <instance_id>           # DELETE + teardown reminders

Hard-won constraints baked in (do not "simplify" them away):
  - Offer search is GET ?q=<urlencoded JSON>; a POST body is silently rejected
    (0 offers, no error).
  - The pip `vastai` CLI does not install on this Mac (Py3.14 pyexpat) — this
    tool is stdlib urllib only.
  - Offer/instance payloads contain raw control characters → json strict=False.
  - Direct SSH endpoint = ports["22/tcp"][0].HostPort + public_ipaddr; the
    ssh_host:ssh_port pair is a proxy that sometimes closes connections. Check
    BOTH before declaring an instance dead.
  - Dud hosts exist: actual_status=running but port 22 never listens. After
    --dud-after seconds we destroy and re-rent the next offer automatically.
  - Deploy to the box via GitHub clone, never rsync of the working tree.
  - Only instances recorded in the local ledger (created by this tool) may be
    destroyed — never touch another agent's box.

Stdlib-only on purpose: runs under any python3, no venv required.
Style: pure functions + frozen dataclasses in the core, fail-fast sys.exit at
the CLI edge (Result only where parsing is genuinely fallible).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.result import Err, Ok, Result

API_BASE = "https://console.vast.ai/api/v0"
KEY_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"
LEDGER_PATH = Path.home() / ".config" / "vastai" / "vast_box_ledger.json"
SSH_CONFIG_PATH = Path.home() / ".ssh" / "config"
# Session-scoped blocklist of Vast machine_ids that failed a race probe (never
# booted / no port 22). `race` skips them so a known-dud machine isn't re-rented.
# Plain text, one id per line; safe to delete any time (it refills itself).
QUARANTINE_PATH = Path.home() / ".config" / "vastai" / "vast_quarantine"

TEMPLATE_ID = 405071  # "PyTorch (Vast)" — /venv/main + sshd baked in
DEFAULT_DISK_GB = 60
DEFAULT_SSH_ALIAS = "vast"
REPO_URL = "https://github.com/jca225/tracklist_engine.git"
REPO_DIR = "/workspace/tracklist_engine"
PI_TS_HOST = "pi-storage.tail116c2d.ts.net"

# The validated offer filter (memory: project_vastai_instance_choice /
# project_vast_access). direct_port_count>0 is the decisive field — boxes
# with 0 may never map SSH.
DEFAULT_QUERY: dict[str, object] = {
    "gpu_name": {"eq": "RTX 4090"},
    "num_gpus": {"eq": 1},
    "rentable": {"eq": True},
    "direct_port_count": {"gt": 0},
    "reliability2": {"gt": 0.99},
    "inet_down": {"gt": 500},
    "inet_up": {"gt": 500},
    "disk_space": {"gt": 40},
    "order": [["dph_total", "asc"]],
    "type": "ask",
}

# (method, url, body) -> (http_status, response_text). Injected in tests.
HttpFn = Callable[[str, str, "bytes | None"], "tuple[int, str]"]
# (host, port) -> port 22 listening?  Injected in tests.
ProbeFn = Callable[[str, int], bool]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Offer:
    offer_id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    dph_total: float
    reliability: float
    inet_down: float
    inet_up: float
    direct_ports: int
    disk_space: float
    geolocation: str


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int


@dataclass(frozen=True)
class Instance:
    instance_id: int
    machine_id: int | None
    label: str | None
    actual_status: str | None
    gpu_name: str | None
    dph_total: float | None
    gpu_util: float | None
    direct: Endpoint | None
    proxy: Endpoint | None


@dataclass(frozen=True)
class WaitState:
    """Dud-detection state machine: booting → running (timer starts) →
    ready | dud. `running_since` is monotonic seconds of the first poll
    that saw actual_status=running."""

    started: float
    running_since: float | None


Verdict = Literal["wait", "ready", "dud", "boot_timeout"]


# ---------------------------------------------------------------------------
# Pure core — query building, parsing, ranking
# ---------------------------------------------------------------------------


def build_query(
    gpu_name: str | None = None, max_dph: float | None = None
) -> dict[str, object]:
    q = dict(DEFAULT_QUERY)
    if gpu_name is not None:
        q["gpu_name"] = {"eq": gpu_name}
    if max_dph is not None:
        q["dph_total"] = {"lt": max_dph}
    return q


def bundles_url(query: dict[str, object]) -> str:
    return f"{API_BASE}/bundles/?q={urllib.parse.quote(json.dumps(query))}"


def _num(raw: object, default: float = 0.0) -> float:
    return float(raw) if isinstance(raw, (int, float)) else default


def parse_offers(text: str) -> Result[tuple[Offer, ...], str]:
    try:
        payload = json.loads(text, strict=False)  # payload has raw control chars
    except json.JSONDecodeError as exc:
        return Err(f"offers response is not JSON: {exc}")
    rows = payload.get("offers") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return Err(f"no 'offers' array in response: {str(payload)[:200]}")
    offers = tuple(
        Offer(
            offer_id=int(r["id"]),
            machine_id=int(r.get("machine_id") or 0),
            gpu_name=str(r.get("gpu_name") or "?"),
            num_gpus=int(r.get("num_gpus") or 1),
            dph_total=_num(r.get("dph_total")),
            reliability=_num(r.get("reliability2")),
            inet_down=_num(r.get("inet_down")),
            inet_up=_num(r.get("inet_up")),
            direct_ports=int(r.get("direct_port_count") or 0),
            disk_space=_num(r.get("disk_space")),
            geolocation=str(r.get("geolocation") or "?"),
        )
        for r in rows
        if isinstance(r, dict) and "id" in r
    )
    return Ok(offers)


def eligible_offers(
    offers: tuple[Offer, ...], exclude_machines: frozenset[int] = frozenset()
) -> tuple[Offer, ...]:
    """Re-assert the constraints the server should have applied (belt and
    braces) and drop machines we already know are duds, cheapest first."""
    ok = [
        o for o in offers if o.direct_ports > 0 and o.machine_id not in exclude_machines
    ]
    return tuple(sorted(ok, key=lambda o: o.dph_total))


def _endpoint_from_ports(raw: dict[str, object]) -> Endpoint | None:
    ports = raw.get("ports")
    ip = raw.get("public_ipaddr")
    if not isinstance(ports, dict) or not isinstance(ip, str) or not ip:
        return None
    mapping = ports.get("22/tcp")
    if not isinstance(mapping, list) or not mapping:
        return None
    first = mapping[0]
    if not isinstance(first, dict):
        return None
    try:
        return Endpoint(host=ip.strip(), port=int(first["HostPort"]))
    except (KeyError, TypeError, ValueError):
        return None


def _proxy_endpoint(raw: dict[str, object]) -> Endpoint | None:
    host, port = raw.get("ssh_host"), raw.get("ssh_port")
    if isinstance(host, str) and host and isinstance(port, (int, float)):
        return Endpoint(host=host, port=int(port))
    return None


def parse_instance(raw: dict[str, object]) -> Instance:
    return Instance(
        instance_id=int(raw["id"]),  # type: ignore[arg-type]
        machine_id=int(raw["machine_id"])
        if isinstance(raw.get("machine_id"), int)
        else None,
        label=raw.get("label") if isinstance(raw.get("label"), str) else None,
        actual_status=raw.get("actual_status")
        if isinstance(raw.get("actual_status"), str)
        else None,
        gpu_name=raw.get("gpu_name") if isinstance(raw.get("gpu_name"), str) else None,
        dph_total=_num(raw.get("dph_total"))
        if raw.get("dph_total") is not None
        else None,
        gpu_util=_num(raw.get("gpu_util")) if raw.get("gpu_util") is not None else None,
        direct=_endpoint_from_ports(raw),
        proxy=_proxy_endpoint(raw),
    )


def parse_instances(text: str) -> Result[tuple[Instance, ...], str]:
    try:
        payload = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        return Err(f"instances response is not JSON: {exc}")
    rows = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return Err(f"no 'instances' array in response: {str(payload)[:200]}")
    return Ok(
        tuple(parse_instance(r) for r in rows if isinstance(r, dict) and "id" in r)
    )


def label_conflict(instances: tuple[Instance, ...], label: str) -> Instance | None:
    for inst in instances:
        if inst.label == label:
            return inst
    return None


# ---------------------------------------------------------------------------
# Pure core — dud-detection state machine
# ---------------------------------------------------------------------------


def advance(
    state: WaitState,
    *,
    status: str | None,
    port_listening: bool,
    now: float,
    dud_after: float = 600.0,
    boot_timeout: float = 1800.0,
) -> tuple[WaitState, Verdict]:
    """One poll step. The dud clock starts at the first running observation:
    a box that reports running but whose port 22 never listens within
    `dud_after` is declared a dud (destroy + re-rent). A box that never even
    reaches running within `boot_timeout` times out the same way."""
    if status == "running":
        running_since = state.running_since if state.running_since is not None else now
        next_state = WaitState(started=state.started, running_since=running_since)
        if port_listening:
            return next_state, "ready"
        if now - running_since >= dud_after:
            return next_state, "dud"
        return next_state, "wait"
    if now - state.started >= boot_timeout:
        return state, "boot_timeout"
    return state, "wait"


def race_step(
    states: dict[int, WaitState],
    observations: dict[int, tuple[str | None, bool]],
    *,
    now: float,
    dud_after: float = 600.0,
    boot_timeout: float = 1800.0,
) -> tuple[dict[int, WaitState], dict[int, Verdict]]:
    """One concurrent poll across a racing pool. Runs the per-box `advance`
    state machine and returns updated states + a verdict per box. The caller
    takes the first ``ready`` as the winner (destroy the rest) and quarantines
    any ``dud``/``boot_timeout`` machine. Pure — every box advances
    independently from its own ``WaitState``."""
    new_states: dict[int, WaitState] = {}
    verdicts: dict[int, Verdict] = {}
    for iid, (status, listening) in observations.items():
        st, v = advance(
            states[iid],
            status=status,
            port_listening=listening,
            now=now,
            dud_after=dud_after,
            boot_timeout=boot_timeout,
        )
        new_states[iid] = st
        verdicts[iid] = v
    return new_states, verdicts


# ---------------------------------------------------------------------------
# Pure core — ~/.ssh/config rewrite
# ---------------------------------------------------------------------------


def ssh_config_block(alias: str, host: str, port: int) -> tuple[str, ...]:
    return (
        f"Host {alias}",
        f"    HostName {host}",
        f"    Port {port}",
        "    User root",
        "    StrictHostKeyChecking accept-new",
        "    UserKnownHostsFile /dev/null",
    )


def rewrite_ssh_config(text: str, alias: str, host: str, port: int) -> Result[str, str]:
    """Replace the single-alias `Host <alias>` block (or append one),
    preserving every other byte of the config."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        tokens = lines[i].split()
        if tokens[:1] == ["Host"]:
            aliases = tokens[1:]
            if alias in aliases and aliases != [alias]:
                return Err(
                    f"'{lines[i].strip()}' covers multiple aliases — refusing to "
                    f"rewrite; split '{alias}' into its own Host block first"
                )
            if aliases == [alias]:
                i += 1
                while i < len(lines) and lines[i].split()[:1] != ["Host"]:
                    i += 1
                if not replaced:
                    out.extend(ssh_config_block(alias, host, port))
                    replaced = True
                continue
        out.append(lines[i])
        i += 1
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.extend(ssh_config_block(alias, host, port))
    return Ok("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Pure core — ownership ledger (destroy-only-yours protocol)
# ---------------------------------------------------------------------------


def ledger_add(
    ledger: dict[str, dict[str, object]],
    instance_id: int,
    label: str,
    disk: int,
    created_utc: float,
) -> dict[str, dict[str, object]]:
    updated = dict(ledger)
    updated[str(instance_id)] = {
        "label": label,
        "disk": disk,
        "created_utc": created_utc,
    }
    return updated


def ledger_remove(
    ledger: dict[str, dict[str, object]], instance_id: int
) -> dict[str, dict[str, object]]:
    updated = dict(ledger)
    updated.pop(str(instance_id), None)
    return updated


def ledger_owns(ledger: dict[str, dict[str, object]], instance_id: int) -> bool:
    return str(instance_id) in ledger


# ---------------------------------------------------------------------------
# Pure core — provisioning plan + pubkey line
# ---------------------------------------------------------------------------


def provision_plan(branch: str, roformer: bool) -> tuple[tuple[str, str], ...]:
    """(description, remote command) steps run over `ssh <alias>`. Deploy is
    GitHub-clone only — never rsync of the working tree (bulk tree uploads
    are classifier-blocked, and the box must run committed code anyway)."""
    steps = [
        (
            "clone repo (GitHub, shallow)",
            f"test -d {REPO_DIR}/.git || git clone --depth 1 --branch {branch} {REPO_URL} {REPO_DIR}",
        ),
        (
            "bootstrap (apt + /venv/main deps + essentia + tailscaled)",
            f"cd {REPO_DIR} && SKIP_CLONE=1 REPO_DIR={REPO_DIR} bash scripts/vast_bootstrap.sh",
        ),
        (
            "bashrc: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch>=2.12 weights_only "
            "breaks beat_this/MSST checkpoint loads)",
            "grep -q TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD ~/.bashrc || "
            "echo 'export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1' >> ~/.bashrc",
        ),
        (
            "verify CUDA",
            '/venv/main/bin/python -c "import torch; '
            "assert torch.cuda.is_available(), 'CUDA unavailable'; "
            'print(torch.__version__, torch.cuda.get_device_name(0))"',
        ),
    ]
    if roformer:
        steps.append(
            (
                "roformer checkpoints (~2.5 GB, background — tail "
                "/workspace/roformer_setup.log)",
                f"cd {REPO_DIR} && nohup bash scripts/setup_roformer_separation.sh "
                "> /workspace/roformer_setup.log 2>&1 & echo roformer setup started",
            )
        )
    return tuple(steps)


def authorized_key_line(pubkey: str, label: str, instance_id: int) -> str:
    """Normalize a `ssh-ed25519 AAAA... root@host` pubkey to the exact line we
    ask the user to append on pi — our comment replaces the box's so teardown
    can find and remove it by `vast-<label>-<id>`."""
    fields = pubkey.split()
    if len(fields) < 2:
        raise ValueError(f"not an ssh public key: {pubkey!r}")
    return f"{fields[0]} {fields[1]} vast-{label}-{instance_id}"


TAILSCALE_URL_RE = re.compile(r"https://login\.tailscale\.com/a/\S+")


# ---------------------------------------------------------------------------
# Edges — HTTP, ledger IO, ssh subprocess (fail-fast)
# ---------------------------------------------------------------------------


def read_api_key() -> str:
    if not KEY_PATH.exists():
        sys.exit(f"vast_box: API key not found at {KEY_PATH}")
    return KEY_PATH.read_text().strip()


def real_http(method: str, url: str, body: bytes | None) -> tuple[int, str]:
    headers = {
        "Authorization": f"Bearer {read_api_key()}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        sys.exit(f"vast_box: cannot reach {url}: {exc.reason}")


def real_probe(host: str, port: int) -> bool:
    r = subprocess.run(
        ["nc", "-z", "-w", "5", host, str(port)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def load_ledger() -> dict[str, dict[str, object]]:
    if not LEDGER_PATH.exists():
        return {}
    data = json.loads(LEDGER_PATH.read_text())
    return data if isinstance(data, dict) else {}


def save_ledger(ledger: dict[str, dict[str, object]]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2) + "\n")


def load_quarantine() -> frozenset[int]:
    """Machine ids the `race` command should skip (failed a recent probe)."""
    if not QUARANTINE_PATH.exists():
        return frozenset()
    return frozenset(
        int(line.strip())
        for line in QUARANTINE_PATH.read_text().splitlines()
        if line.strip().isdigit()
    )


def add_quarantine(machine_ids: set[int]) -> None:
    ids = {m for m in machine_ids if m}
    if not ids:
        return
    merged = set(load_quarantine()) | ids
    QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUARANTINE_PATH.write_text("\n".join(str(m) for m in sorted(merged)) + "\n")


def ssh_run(
    alias: str, remote_cmd: str, *, timeout: int = 1800
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", alias, remote_cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def ssh_run_or_die(
    alias: str, desc: str, remote_cmd: str, *, timeout: int = 1800
) -> str:
    print(f"==> {desc}")
    r = ssh_run(alias, remote_cmd, timeout=timeout)
    if r.returncode != 0:
        sys.exit(
            f"vast_box: step failed: {desc}\n  cmd: {remote_cmd}\n"
            f"  stdout: {r.stdout.strip()[-2000:]}\n  stderr: {r.stderr.strip()[-2000:]}"
        )
    return r.stdout


def fetch_offers(http: HttpFn, query: dict[str, object]) -> tuple[Offer, ...]:
    status, text = http("GET", bundles_url(query), None)
    if status != 200:
        sys.exit(f"vast_box: offer search failed (HTTP {status}): {text[:300]}")
    match parse_offers(text):
        case Ok(offers):
            return offers
        case Err(msg):
            sys.exit(f"vast_box: {msg}")
    raise AssertionError("unreachable")


def fetch_instances(http: HttpFn) -> tuple[Instance, ...]:
    status, text = http("GET", f"{API_BASE}/instances/", None)
    if status != 200:
        sys.exit(f"vast_box: instance list failed (HTTP {status}): {text[:300]}")
    match parse_instances(text):
        case Ok(instances):
            return instances
        case Err(msg):
            sys.exit(f"vast_box: {msg}")
    raise AssertionError("unreachable")


def find_instance(instances: tuple[Instance, ...], instance_id: int) -> Instance | None:
    for inst in instances:
        if inst.instance_id == instance_id:
            return inst
    return None


def rent_offer(http: HttpFn, offer_id: int, label: str, disk: int) -> int:
    body = json.dumps(
        {"template_id": TEMPLATE_ID, "disk": disk, "label": label}
    ).encode()
    status, text = http("PUT", f"{API_BASE}/asks/{offer_id}/", body)
    if status != 200:
        sys.exit(f"vast_box: rent failed (HTTP {status}): {text[:300]}")
    payload = json.loads(text, strict=False)
    if not payload.get("success"):
        sys.exit(f"vast_box: rent rejected: {text[:300]}")
    new_id = payload.get("new_contract")
    if not isinstance(new_id, int):
        sys.exit(
            f"vast_box: rent succeeded but no new_contract in response: {text[:300]}"
        )
    ledger = ledger_add(load_ledger(), new_id, label, disk, time.time())
    save_ledger(ledger)
    return new_id


def destroy_instance(http: HttpFn, instance_id: int) -> None:
    status, text = http("DELETE", f"{API_BASE}/instances/{instance_id}/", None)
    if status != 200:
        sys.exit(f"vast_box: destroy failed (HTTP {status}): {text[:300]}")
    save_ledger(ledger_remove(load_ledger(), instance_id))


def print_teardown_reminder(label: str, instance_id: int) -> None:
    print(
        "\nTeardown protocol — the box's pubkey is still on pi-storage. Remove it"
        " (USER runs this; prod authorized_keys edits are yours, not the agent's):\n"
        f"    ssh pi-storage 'sed -i \"/vast-{label}-{instance_id}/d\" ~/.ssh/authorized_keys'\n"
        "Also remove the node from the tailnet admin console if it was linked:"
        " https://login.tailscale.com/admin/machines"
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_search(args: argparse.Namespace, http: HttpFn) -> None:
    query = build_query(gpu_name=args.gpu, max_dph=args.max_dph)
    offers = eligible_offers(fetch_offers(http, query))
    if not offers:
        sys.exit("vast_box: no eligible offers — relax --gpu / --max-dph and retry")
    print(f"{len(offers)} eligible offers (cheapest first):\n")
    print(
        f"{'OFFER_ID':>10}  {'GPU':<14} {'$/hr':>6} {'rel':>6} {'down':>6} {'up':>6} {'disk':>6}  geo"
    )
    for o in offers[: args.limit]:
        print(
            f"{o.offer_id:>10}  {o.gpu_name:<14} {o.dph_total:>6.3f} {o.reliability:>6.3f}"
            f" {o.inet_down:>6.0f} {o.inet_up:>6.0f} {o.disk_space:>6.0f}  {o.geolocation}"
        )
    print(
        f"\nNext: venvs/audio/bin/python scripts/vast_box.py rent {offers[0].offer_id} --label <purpose>"
    )


def cmd_rent(args: argparse.Namespace, http: HttpFn) -> None:
    # Protocol: LIST BEFORE CREATE — show what's already running, refuse a
    # duplicate label (one distinctly-labeled box per agent).
    instances = fetch_instances(http)
    if instances:
        print("Existing instances (do NOT touch ones you didn't create):")
        for inst in instances:
            print(
                f"  {inst.instance_id}  label={inst.label!r} status={inst.actual_status}"
                f" gpu={inst.gpu_name} ${inst.dph_total or 0:.3f}/hr gpu_util={inst.gpu_util}"
            )
    else:
        print("No instances currently running on the account.")
    if (dup := label_conflict(instances, args.label)) is not None:
        sys.exit(
            f"vast_box: label {args.label!r} already in use by instance "
            f"{dup.instance_id} — one box per label; pick another or destroy yours first"
        )
    new_id = rent_offer(http, args.offer_id, args.label, args.disk)
    print(
        f"\nRented instance {new_id} (label={args.label!r}, template {TEMPLATE_ID},"
        f" disk {args.disk} GB) — recorded in {LEDGER_PATH}."
    )
    print(f"Next: venvs/audio/bin/python scripts/vast_box.py wait-ssh {new_id}")


def _apply_ssh_config(alias: str, endpoint: Endpoint) -> None:
    text = SSH_CONFIG_PATH.read_text() if SSH_CONFIG_PATH.exists() else ""
    match rewrite_ssh_config(text, alias, endpoint.host, endpoint.port):
        case Ok(updated):
            SSH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            SSH_CONFIG_PATH.write_text(updated)
            print(f"~/.ssh/config: Host {alias} -> {endpoint.host}:{endpoint.port}")
        case Err(msg):
            sys.exit(f"vast_box: ssh config rewrite refused: {msg}")


def _wait_one(
    http: HttpFn,
    probe: ProbeFn,
    instance_id: int,
    *,
    poll_s: float,
    dud_after: float,
    boot_timeout: float,
) -> tuple[Verdict, Instance | None]:
    state = WaitState(started=time.monotonic(), running_since=None)
    while True:
        inst = find_instance(fetch_instances(http), instance_id)
        if inst is None:
            sys.exit(f"vast_box: instance {instance_id} not found on the account")
        listening = inst.direct is not None and probe(
            inst.direct.host, inst.direct.port
        )
        state, verdict = advance(
            state,
            status=inst.actual_status,
            port_listening=listening,
            now=time.monotonic(),
            dud_after=dud_after,
            boot_timeout=boot_timeout,
        )
        direct = f"{inst.direct.host}:{inst.direct.port}" if inst.direct else "-"
        proxy = f"{inst.proxy.host}:{inst.proxy.port}" if inst.proxy else "-"
        print(
            f"  status={inst.actual_status or 'none'} direct={direct}"
            f" (listening={listening}) proxy={proxy} -> {verdict}"
        )
        if verdict != "wait":
            return verdict, inst
        time.sleep(poll_s)


def cmd_wait_ssh(args: argparse.Namespace, http: HttpFn, probe: ProbeFn) -> None:
    instance_id = args.instance_id
    dud_machines: set[int] = set()
    for attempt in range(args.max_rerents + 1):
        print(f"==> waiting for instance {instance_id} (attempt {attempt + 1})")
        verdict, inst = _wait_one(
            http,
            probe,
            instance_id,
            poll_s=args.poll,
            dud_after=args.dud_after,
            boot_timeout=args.boot_timeout,
        )
        if verdict == "ready" and inst is not None and inst.direct is not None:
            _apply_ssh_config(args.alias, inst.direct)
            if inst.proxy:
                print(
                    f"(proxy fallback if direct ever flakes: root@{inst.proxy.host} -p {inst.proxy.port})"
                )
            print(f"Ready: ssh {args.alias}")
            return
        # Dud host / boot timeout: destroy and re-rent the next offer.
        ledger_entry = load_ledger().get(str(instance_id), {})
        label = str(ledger_entry.get("label") or "unknown")
        disk = int(ledger_entry.get("disk") or DEFAULT_DISK_GB)  # type: ignore[arg-type]
        if inst is not None and inst.machine_id is not None:
            dud_machines.add(inst.machine_id)
        print(
            f"==> {verdict}: destroying instance {instance_id} (host never opened port 22)"
        )
        destroy_instance(http, instance_id)
        if attempt >= args.max_rerents:
            break
        offers = eligible_offers(
            fetch_offers(http, build_query()), frozenset(dud_machines)
        )
        if not offers:
            sys.exit("vast_box: dud destroyed but no replacement offers available")
        print(
            f"==> re-renting next offer {offers[0].offer_id} (${offers[0].dph_total:.3f}/hr)"
        )
        instance_id = rent_offer(http, offers[0].offer_id, label, disk)
    sys.exit(
        f"vast_box: gave up after {args.max_rerents + 1} dud hosts — try a different --gpu or later"
    )


def cmd_race(args: argparse.Namespace, http: HttpFn, probe: ProbeFn) -> None:
    """Rent N boxes at once, keep the first to open port 22, destroy the losers
    and quarantine any dud machine. Faster and more robust than the sequential
    wait-ssh dud re-rent: you don't discover a bad box only after staging code
    on it. (Port-22-listening is the health signal — it catches the common
    never-boot/no-map dud; a deeper post-SSH GPU/CUDA probe is a future add.)"""
    if args.n < 1:
        sys.exit("vast_box: --n must be >= 1")
    query = build_query(args.gpu, args.max_dph)
    active: dict[int, int] = {}  # instance_id -> machine_id (only boxes WE rented)

    def destroy_all(ids: list[int]) -> None:
        for iid in ids:  # KEEP-guard: only ids in `active`, which we rented
            try:
                destroy_instance(http, iid)
            except SystemExit:
                pass  # already gone / API hiccup — don't abort the sweep

    for rnd in range(args.rounds):
        offers = eligible_offers(fetch_offers(http, query), load_quarantine())
        need = args.n - len(active)
        if need > 0:
            if len(offers) < need and not active:
                sys.exit(
                    f"vast_box: only {len(offers)} eligible offers (need {need}) — "
                    "widen --gpu/--max-dph or clear the quarantine"
                )
            for off in offers[:need]:
                iid = rent_offer(http, off.offer_id, args.label, args.disk)
                active[iid] = off.machine_id
                print(
                    f"==> race round {rnd + 1}: rented {iid} "
                    f"(machine {off.machine_id}, ${off.dph_total:.3f}/hr, {off.geolocation})"
                )
        states = {
            iid: WaitState(started=time.monotonic(), running_since=None)
            for iid in active
        }
        while active:
            instances = fetch_instances(http)
            obs: dict[int, tuple[str | None, bool]] = {}
            for iid in active:
                inst = find_instance(instances, iid)
                listening = bool(
                    inst is not None
                    and inst.direct is not None
                    and probe(inst.direct.host, inst.direct.port)
                )
                obs[iid] = (inst.actual_status if inst else None, listening)
            states, verdicts = race_step(
                states,
                obs,
                now=time.monotonic(),
                dud_after=args.dud_after,
                boot_timeout=args.boot_timeout,
            )
            print("  " + "  ".join(f"{iid}:{verdicts[iid]}" for iid in sorted(active)))
            winner = next((iid for iid, v in verdicts.items() if v == "ready"), None)
            if winner is not None:
                losers = [iid for iid in active if iid != winner]
                print(f"==> WINNER {winner} — destroying {len(losers)} loser(s)")
                destroy_all(losers)
                inst = find_instance(fetch_instances(http), winner)
                if inst is not None and inst.direct is not None:
                    _apply_ssh_config(args.alias, inst.direct)
                    if inst.proxy:
                        print(
                            f"(proxy fallback: root@{inst.proxy.host} -p {inst.proxy.port})"
                        )
                print(f"Ready: ssh {args.alias}")
                print(f"Winner instance id: {winner}")
                return
            duds = {iid for iid, v in verdicts.items() if v in ("dud", "boot_timeout")}
            if duds:
                add_quarantine({active[iid] for iid in duds})
                print(
                    f"==> {len(duds)} dud(s) {sorted(duds)} — destroyed + machines quarantined"
                )
                destroy_all(list(duds))
                for iid in duds:
                    active.pop(iid, None)
                    states.pop(iid, None)
            if not active:
                break  # whole pool died — outer loop re-rents a fresh batch
            time.sleep(args.poll)
    sys.exit(f"vast_box: no healthy box after {args.rounds} race round(s)")


def cmd_provision(args: argparse.Namespace) -> None:
    for desc, remote_cmd in provision_plan(args.branch, args.roformer):
        out = ssh_run_or_die(args.alias, desc, remote_cmd)
        tail = out.strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line}")
    print(
        "\nProvisioned. Next: venvs/audio/bin/python scripts/vast_box.py link-pi"
        f" --id {args.id if args.id else '<instance_id>'}"
    )


def _sole_ledger_id() -> int | None:
    ledger = load_ledger()
    if len(ledger) == 1:
        return int(next(iter(ledger)))
    return None


def cmd_link_pi(args: argparse.Namespace) -> None:
    instance_id = args.id if args.id is not None else _sole_ledger_id()
    if instance_id is None:
        sys.exit("vast_box: pass --id <instance_id> (ledger has 0 or >1 entries)")
    ledger_entry = load_ledger().get(str(instance_id), {})
    label = args.label or str(ledger_entry.get("label") or "").strip()
    if not label:
        sys.exit("vast_box: pass --label (instance not in ledger)")

    pubkey = ssh_run_or_die(
        args.alias,
        "ensure box ed25519 key",
        'test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 >/dev/null; '
        "cat ~/.ssh/id_ed25519.pub",
    ).strip()
    key_line = authorized_key_line(pubkey, label, instance_id)

    # tailscaled (userspace, socks5 on :1055) is started by the bootstrap; be
    # defensive in case this runs on a box bootstrapped long ago.
    ssh_run_or_die(
        args.alias,
        "ensure tailscaled (userspace) is running",
        "pgrep -x tailscaled >/dev/null || { mkdir -p /var/run/tailscale; "
        "nohup tailscaled --tun=userspace-networking --socks5-server=localhost:1055 "
        "--outbound-http-proxy-listen=localhost:1055 > /var/log/tailscaled.log 2>&1 & sleep 2; }",
    )
    ts_out = ssh_run_or_die(
        args.alias,
        f"tailscale up --hostname=vast-{label} (captures login URL)",
        f"rm -f /tmp/ts_up.log; nohup tailscale up --hostname=vast-{label} > /tmp/ts_up.log 2>&1 & "
        "sleep 8; cat /tmp/ts_up.log; tailscale status --peers=false 2>&1 | head -2 || true",
    )
    url_match = TAILSCALE_URL_RE.search(ts_out)

    print("\n" + "=" * 72)
    print(
        "TWO HUMAN STEPS — the agent cannot do these (prod key append + browser auth):"
    )
    print("\n(a) Authorize the box's key on pi-storage — run on YOUR Mac:\n")
    print(f"    ssh pi-storage 'echo \"{key_line}\" >> ~/.ssh/authorized_keys'")
    print(
        "\n(b) Authenticate the box on the tailnet — click this URL in your browser:\n"
    )
    if url_match:
        print(f"    {url_match.group(0)}")
    else:
        print(
            "    (no login URL captured — the box may already be logged in; check below)"
        )
        print("    " + ts_out.strip().replace("\n", "\n    "))
    print("\n" + "=" * 72)
    input("\nPress Enter once BOTH steps are done (Ctrl-C to abort)... ")

    print("==> verifying box -> pi-storage over the tailnet")
    for attempt in range(6):
        r = ssh_run(
            args.alias,
            f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 {PI_TS_HOST} true",
            timeout=120,
        )
        if r.returncode == 0:
            print(
                f"Linked: box reaches {PI_TS_HOST}. Ready for vast_loop.py / vast_worker."
            )
            return
        print(
            f"  not yet ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no route'}); retrying..."
        )
        time.sleep(10)
    sys.exit(
        "vast_box: box still can't reach pi-storage — check `tailscale status` on the box"
        " and the authorized_keys line on pi"
    )


def cmd_destroy(args: argparse.Namespace, http: HttpFn) -> None:
    ledger = load_ledger()
    if not ledger_owns(ledger, args.instance_id):
        sys.exit(
            f"vast_box: instance {args.instance_id} is not in this tool's ledger "
            f"({LEDGER_PATH}) — it may be another agent's box mid-job. Protocol:"
            " destroy only boxes you created (docs/vast_coordination.md). If it's"
            " a confirmed idle orphan, surface it to the user instead."
        )
    label = str(ledger[str(args.instance_id)].get("label") or "unknown")
    inst = find_instance(fetch_instances(http), args.instance_id)
    if inst is not None:
        print(
            f"Destroying {args.instance_id} label={inst.label!r} status={inst.actual_status}"
            f" gpu_util={inst.gpu_util} ${inst.dph_total or 0:.3f}/hr"
        )
    destroy_instance(http, args.instance_id)
    print(f"Destroyed {args.instance_id} — billing stopped (destroy, not stop).")
    print_teardown_reminder(label, args.instance_id)


# ---------------------------------------------------------------------------
# CLI edge
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vast_box.py",
        description="Vast.ai GPU-box lifecycle: search -> rent -> wait-ssh -> provision -> link-pi -> destroy",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("search", help="list eligible GPU offers, cheapest first")
    sp.add_argument("--gpu", default=None, help='GPU name filter (default "RTX 4090")')
    sp.add_argument("--max-dph", type=float, default=None, help="max $/hr")
    sp.add_argument("--limit", type=int, default=15)

    rp = sub.add_parser("rent", help="rent an offer (template 405071, ledger-tracked)")
    rp.add_argument("offer_id", type=int)
    rp.add_argument("--label", required=True, help="purpose label, e.g. mert-backfill")
    rp.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)

    wp = sub.add_parser(
        "wait-ssh", help="poll until SSH is reachable; dud boxes auto-re-rent"
    )
    wp.add_argument("instance_id", type=int)
    wp.add_argument("--alias", default=DEFAULT_SSH_ALIAS)
    wp.add_argument("--poll", type=float, default=15.0)
    wp.add_argument("--dud-after", type=float, default=600.0)
    wp.add_argument("--boot-timeout", type=float, default=1800.0)
    wp.add_argument("--max-rerents", type=int, default=2)

    rc = sub.add_parser(
        "race",
        help="rent N boxes at once, keep the first healthy one, destroy+quarantine duds",
    )
    rc.add_argument("--label", required=True, help="purpose label (shared by the pool)")
    rc.add_argument("--n", type=int, default=3, help="boxes to race (default 3)")
    rc.add_argument("--gpu", default=None, help='GPU name filter (default "RTX 4090")')
    rc.add_argument("--max-dph", type=float, default=None, help="max $/hr")
    rc.add_argument("--disk", type=int, default=DEFAULT_DISK_GB)
    rc.add_argument("--alias", default=DEFAULT_SSH_ALIAS)
    rc.add_argument("--poll", type=float, default=15.0)
    rc.add_argument("--dud-after", type=float, default=600.0)
    rc.add_argument("--boot-timeout", type=float, default=1800.0)
    rc.add_argument(
        "--rounds", type=int, default=2, help="re-rent rounds if a whole pool duds"
    )

    pp = sub.add_parser("provision", help="GitHub clone + vast_bootstrap.sh on the box")
    pp.add_argument("--alias", default=DEFAULT_SSH_ALIAS)
    pp.add_argument("--branch", default="main")
    pp.add_argument(
        "--roformer",
        action="store_true",
        help="also fetch RoFormer checkpoints (2.5 GB, background)",
    )
    pp.add_argument(
        "--id", type=int, default=None, help="instance id (for the next-step hint)"
    )

    lp = sub.add_parser(
        "link-pi", help="box key + tailscale link to pi-storage (2 human steps)"
    )
    lp.add_argument("--alias", default=DEFAULT_SSH_ALIAS)
    lp.add_argument(
        "--id",
        type=int,
        default=None,
        help="instance id (defaults to sole ledger entry)",
    )
    lp.add_argument("--label", default=None, help="override label (defaults to ledger)")

    dp = sub.add_parser("destroy", help="destroy YOUR labeled box (refuses others)")
    dp.add_argument("instance_id", type=int)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "search":
        cmd_search(args, real_http)
    elif args.cmd == "rent":
        cmd_rent(args, real_http)
    elif args.cmd == "wait-ssh":
        cmd_wait_ssh(args, real_http, real_probe)
    elif args.cmd == "race":
        cmd_race(args, real_http, real_probe)
    elif args.cmd == "provision":
        cmd_provision(args)
    elif args.cmd == "link-pi":
        cmd_link_pi(args)
    elif args.cmd == "destroy":
        cmd_destroy(args, real_http)


if __name__ == "__main__":
    main()
