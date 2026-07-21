"""gpubox driver: BB11 structure_cue agentic + BB10 pool agentic on one 4090.

Uses gpubox Box.launch (NOT raw Vast curl). Ownership from Registry + EventLog.
Jobs run detached under tmux; this process does NOT destroy the box on exit —
attach later, pull timelines, then box.destroy() with positive ownership.

    cd ~/workspace/gpubox && .venv/bin/python \
      /Users/johnnycabrahams/Desktop/tracklist_engine/scripts/gpubox_agentic_both.py

Monitor:
    cd ~/workspace/gpubox && .venv/bin/python - <<'PY'
    from gpubox.api import Box
    b = Box.attach("align-agentic-both")
    print(b.logs("agentic-both", tail=80) if b else "gone")
    PY
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "workspace/gpubox/src"))

from gpubox.api import Box  # noqa: E402
from gpubox.client import VastClient, read_api_key  # noqa: E402
from gpubox.models import Spec  # noqa: E402
from gpubox.registry import Registry  # noqa: E402

HOME = Path.home() / ".gpubox"
REPO = Path("/Users/johnnycabrahams/Desktop/tracklist_engine")
RUN_BOTH = "align-agentic-both"
RUN_BB10 = "align-bb10-flywheel"
MAX_DPH = 0.50
CODE = "/workspace/tracklist_engine"
ALIGN_REMOTE = "/workspace/aligning"
# DEFAULT_IMAGE is py3.11; repo needs 3.12+ (`type` aliases). Provision creates this env.
PY = "/opt/conda/envs/align/bin/python"


def _align_dir(set_id: str) -> Path:
    hits = sorted((Path.home() / "aligning").glob(f"{set_id}__*"))
    if not hits:
        raise FileNotFoundError(f"no ~/aligning/{set_id}__*")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bb10-only",
        action="store_true",
        help="sensor-restore re-harvest: sync+run BB10 flywheel only (skip BB11)",
    )
    args = ap.parse_args()
    run = RUN_BB10 if args.bb10_only else RUN_BOTH
    set_ids = ("w1mgcjt",) if args.bb10_only else ("2nvzlh2k", "w1mgcjt")
    purpose = (
        "BB10 flywheel re-harvest after vocal path remap"
        if args.bb10_only
        else "BB11 structure_cue agentic + BB10 unlabeled pool agentic"
    )

    client = VastClient(read_api_key())
    registry = Registry(HOME / "registry.json")
    preexisting = {i.instance_id for i in client.list_instances()}
    print(f"[safety] preexisting (untouched): {sorted(preexisting)}")
    print(f"[mode] run={run} sets={set_ids}")

    # Resume if this run already claimed a box (e.g. after a push-API fix).
    # Prefer newest claim: a killed race can leave an older same-run orphan.
    matches = sorted(
        (
            (e.get("created_at", 0.0), iid)
            for iid, e in registry.entries().items()
            if e.get("run") == run
        ),
        reverse=True,
    )
    box = None
    if matches:
        want_id = matches[0][1]
        box = Box.attach(run, client=client, registry=registry)
        if box is not None and box.instance_id != want_id:
            # Box.attach returns first match; rebuild for newest.
            inst = client.show(want_id)
            if inst is not None:
                from gpubox.session import Session

                box = Box(inst, registry=registry, client=client, session=Session(inst))
            else:
                box = None
        # Destroy older same-run orphans we claim (never touch other runs).
        for _, oid in matches[1:]:
            print(f"[cleanup] destroy stale same-run orphan id={oid}")
            client.destroy(oid)
            registry.release(oid)
    if box is not None:
        print(f"[attach] resume run={run} id={box.instance_id}")
    else:
        spec = Spec(
            gpu="RTX 4090",
            vram_gb=24,
            disk_gb=120,
            cuda=12.4,
            max_dph=MAX_DPH,
            min_inet_down=200.0,  # audio sync; relax vs training default
        )
        t0 = time.time()
        print(f"[launch] run={run} spec={spec}")
        box = Box.launch(
            spec,
            run=run,
            purpose=purpose,
            race_n=2,
            client=client,
            registry=registry,
        )
        print(
            f"[launch] ok id={box.instance_id} gpu={box.instance.gpu_name} "
            f"dph={box.instance.dph_total} in {time.time() - t0:.0f}s"
        )

    # --- provision: system deps + audio pip into DEFAULT python -------------
    # Box.push is thin (local, remote only). Use session.push for timeout/excludes.
    print("[provision] apt…")
    box.session.exec(
        "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "ffmpeg libsndfile1 git rsync curl ca-certificates sqlite3 tmux "
        "build-essential pkg-config >/dev/null",
        timeout=300,
    )
    # Minimal code sync — NEVER mirror the whole Mac checkout (.claude alone ~40G).
    print("[push] repo (allowlist)…")
    box.session.exec(f"mkdir -p {CODE}", timeout=30)
    for rel in (
        "core",
        "analysis",
        "labeling",  # ground_truth + fixtures (infer→dataset imports schema)
        "eda/__init__.py",
        "eda/alignment/__init__.py",
        "eda/alignment/mert_vectors.py",  # mert_store.probe_vector
        "workspaces/section_hsmm",  # stem_placement → similarity_probe._hubert
        "workspaces/alignment_prototype",
        "scripts/vast_bootstrap.sh",
    ):
        src = REPO / rel
        if not src.exists():
            print(f"  skip missing {rel}")
            continue
        dest = f"{CODE}/{rel}"
        box.session.exec(
            f"mkdir -p {dest if src.is_dir() else str(Path(dest).parent)}", timeout=30
        )
        print(f"  push {rel}…")
        excludes: tuple[str, ...] = (
            ".feat_cache",
            "out",
            "experiments/cache",
            "__pycache__",
            "*.pyc",
            ".cache",
        )
        box.session.push(
            str(src) + ("/" if src.is_dir() else ""),
            dest + ("/" if src.is_dir() else ""),
            timeout=900,
            delete=src.is_dir(),
            excludes=excludes if src.is_dir() else (),
        )
    box.session.exec(
        f"mkdir -p {CODE}/workspaces/alignment_prototype/.cache "
        f"{CODE}/workspaces/alignment_prototype/out",
        timeout=60,
    )

    print("[provision] conda env align (py3.12) + torch…")
    box.session.exec(
        f"test -x {PY} || mamba create -y -n align python=3.12 pip >/dev/null",
        timeout=600,
    )
    box.session.exec(
        f"{PY} -m pip install -q --upgrade pip && "
        f"{PY} -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124 && "
        f"{PY} -m pip install -q numpy scipy librosa soundfile pyyaml msgspec "
        "transformers accelerate huggingface_hub einops "
        "sentencepiece protobuf 2>&1 | tail -20",
        timeout=1200,
    )
    rc, out, err = box.session.exec(
        f'cd {CODE} && PYTHONPATH={CODE} {PY} -c "'
        "import torch; from core.result import Ok; "
        "print('CUDA', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); "
        'print(Ok(1))"',
        timeout=60,
    )
    print(out or err)
    if rc != 0 or "CUDA True" not in (out or ""):
        raise RuntimeError(f"CUDA check failed: rc={rc} out={out!r} err={err!r}")

    # --- sync caches + slim aligning (BB11 + BB10) --------------------------
    print("[push] mert/lyrics/fp_index/slot_cues caches…")
    cache = REPO / "workspaces/alignment_prototype/.cache"
    box.session.exec(
        f"mkdir -p {CODE}/workspaces/alignment_prototype/.cache/mert "
        f"{CODE}/workspaces/alignment_prototype/.cache/lyrics "
        f"{CODE}/workspaces/alignment_prototype/.cache/fp_index "
        f"{CODE}/workspaces/alignment_prototype/.cache/slot_cues",
        timeout=30,
    )
    for name in ("mert", "lyrics", "fp_index", "slot_cues"):
        src = cache / name
        if not src.is_dir():
            print(f"  skip missing cache/{name}")
            continue
        print(f"  push cache/{name}…")
        box.session.push(
            str(src) + "/",
            f"{CODE}/workspaces/alignment_prototype/.cache/{name}/",
            timeout=1800,
            delete=False,
        )

    for sid in set_ids:
        local = _align_dir(sid)
        # Space-free remote dir: openrsync truncates host:path at spaces.
        remote = f"{ALIGN_REMOTE}/{sid}/"
        print(f"[push] aligning {sid} slim → {remote}")
        box.session.exec(f"mkdir -p {ALIGN_REMOTE}", timeout=30)
        from gpubox.session import SSH_OPTS, ssh_target

        host, port = ssh_target(box.instance)
        import subprocess

        argv = [
            "rsync",
            "-az",
            "--progress",  # macOS openrsync: no --info=progress2
            "--exclude=candidates/",
            "--exclude=*.asd",
            "--exclude=_stale*",
            "--exclude=*.seedbak",
            "--exclude=_backups/",
            "-e",
            " ".join(["ssh", *SSH_OPTS, "-p", str(port)]),
            str(local) + "/",
            f"root@{host}:{remote}",
        ]
        print(" ", " ".join(argv[:6]), "…")
        subprocess.run(argv, check=True, timeout=3600)

    # find_aligning_dir globs ~/aligning/{set_id}__* — symlink short remote dirs.
    sid_loop = " ".join(set_ids)
    box.session.exec(
        "mkdir -p /root/aligning && "
        f"for sid in {sid_loop}; do "
        f"  src={ALIGN_REMOTE}/$sid; "
        f'  [ -d "$src" ] || src=$(ls -d {ALIGN_REMOTE}/${{sid}}__* 2>/dev/null | head -1); '
        '  [ -n "$src" ] && ln -sfn "$src" /root/aligning/${sid}__vast; '
        "done; ls -la /root/aligning",
        timeout=30,
    )

    # Fail closed if Mac-absolute manifest paths cannot remap (lyrics/hubert starve).
    print("[preflight] mix_vocals + resolvable acappella vocals…")
    sid_list = ", ".join(repr(s) for s in set_ids)
    rc, out, err = box.session.exec(
        f"cd {CODE} && PYTHONPATH={CODE} {PY} - <<'PY'\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "from workspaces.alignment_prototype.infer import _vocal_ref_path\n"
        "root = Path('/root/aligning')\n"
        "ok = True\n"
        f"for sid in [{sid_list}]:\n"
        "    hits = sorted(root.glob(f'{sid}__*'))\n"
        "    if not hits:\n"
        "        # space-free sync lands at /workspace/aligning/<sid>\n"
        "        alt = Path('/workspace/aligning') / sid\n"
        "        hits = [alt] if alt.is_dir() else []\n"
        "    if not hits:\n"
        "        print(f'FAIL {sid}: no aligning dir'); ok = False; continue\n"
        "    d = hits[0]\n"
        "    mixv = d / 'mix_vocals.flac'\n"
        "    if not mixv.is_file():\n"
        "        print(f'FAIL {sid}: missing mix_vocals.flac'); ok = False\n"
        "    tracks = json.loads((d / 'manifest.json').read_text()).get('tracks') or []\n"
        "    n = 0\n"
        "    for t in tracks:\n"
        "        if t.get('stem') != 'acappella' and not (t.get('stems') or {}).get('vocals'):\n"
        "            continue\n"
        "        v = _vocal_ref_path(t, set_dir=d)\n"
        "        if v and Path(v).is_file():\n"
        "            n += 1\n"
        "    print(f'{sid}: dir={d} mix_vocals={mixv.is_file()} resolved_vocals={n}')\n"
        "    if n < 1:\n"
        "        print(f'FAIL {sid}: zero resolvable vocals — lyrics/hubert will abstain')\n"
        "        ok = False\n"
        "sys.exit(0 if ok else 1)\n"
        "PY",
        timeout=120,
    )
    print(out or err)
    if rc != 0:
        raise RuntimeError(
            f"aligning vocal preflight failed rc={rc} — fix path sync/remap before detach"
        )

    # Need predicted timelines on the box
    print("[push] predicted timelines…")
    out_dir = REPO / "workspaces/alignment_prototype/out"
    box.session.exec(f"mkdir -p {CODE}/workspaces/alignment_prototype/out", timeout=30)
    timeline_names = (
        ["w1mgcjt_predicted_timeline.json"]
        if args.bb10_only
        else [
            "2nvzlh2k_predicted_timeline.json",
            "w1mgcjt_predicted_timeline.json",
        ]
    )
    for name in timeline_names:
        p = out_dir / name
        if p.is_file():
            box.session.push(
                str(p),
                f"{CODE}/workspaces/alignment_prototype/out/{name}",
                timeout=120,
                delete=False,
            )

    # Write job script then bash it — run_cmd cannot execute multi-line shell.
    if args.bb10_only:
        job = f"""#!/bin/bash
set -euo pipefail
cd {CODE}
export PYTHONPATH={CODE}
export HF_HOME=/workspace/hf
export TRANSFORMERS_CACHE=/workspace/hf
export PYTHONUNBUFFERED=1
mkdir -p /workspace/hf workspaces/alignment_prototype/out

echo "=== BB10 pool agentic (unlabeled) ==="
{PY} -u -m workspaces.alignment_prototype.drivers.flywheel \\
  --set-id w1mgcjt \\
  --reuse-base workspaces/alignment_prototype/out/w1mgcjt_predicted_timeline.json

echo "=== materialize BB10 pseudo-GT ==="
{PY} -u -m workspaces.alignment_prototype.trajectory.materialize_pseudo_gt \\
  --timeline workspaces/alignment_prototype/out/w1mgcjt_agentic_timeline.json \\
  --out workspaces/alignment_prototype/out/w1mgcjt_pseudo_gt.yaml \\
  --set-id w1mgcjt

echo "ALL_DONE"
date
"""
    else:
        job = f"""#!/bin/bash
set -euo pipefail
cd {CODE}
export PYTHONPATH={CODE}
export HF_HOME=/workspace/hf
export TRANSFORMERS_CACHE=/workspace/hf
export PYTHONUNBUFFERED=1
mkdir -p /workspace/hf workspaces/alignment_prototype/out

echo "=== BB11 structure_cue agentic ==="
{PY} -u - <<'PY'
from pathlib import Path
from workspaces.alignment_prototype.drivers.base import SetContext
from workspaces.alignment_prototype.drivers.agentic import AgenticDriver
ctx = SetContext.for_set("2nvzlh2k", preflight=True)
base = Path("workspaces/alignment_prototype/out/2nvzlh2k_predicted_timeline.json")
if not base.is_file():
    cands = sorted(Path("workspaces/alignment_prototype/out").glob("2nvzlh2k*predicted*timeline*.json"))
    if not cands:
        raise SystemExit("no BB11 base timeline on box — push out/ first")
    base = cands[0]
print("base", base)
out = AgenticDriver(base, live=True).align_set(ctx)
print("BB11_DONE", out)
PY

echo "=== BB10 pool agentic (unlabeled) ==="
{PY} -u -m workspaces.alignment_prototype.drivers.flywheel \\
  --set-id w1mgcjt \\
  --reuse-base workspaces/alignment_prototype/out/w1mgcjt_predicted_timeline.json

echo "=== materialize BB10 pseudo-GT ==="
{PY} -u -m workspaces.alignment_prototype.trajectory.materialize_pseudo_gt \\
  --timeline workspaces/alignment_prototype/out/w1mgcjt_agentic_timeline.json \\
  --out workspaces/alignment_prototype/out/w1mgcjt_pseudo_gt.yaml \\
  --set-id w1mgcjt

echo "ALL_DONE"
date
"""
    print("[run] write + detach agentic-both…")
    # Avoid nesting heredocs in the SSH command — base64 the script.
    import base64

    b64 = base64.b64encode(job.encode()).decode()
    box.session.exec(
        f"echo {b64} | base64 -d > /workspace/agentic_both.sh && chmod +x /workspace/agentic_both.sh",
        timeout=30,
    )
    box.run_cmd(
        "bash /workspace/agentic_both.sh",
        name="agentic-both",
        log="/workspace/agentic_both.log",
    )
    box.heartbeat()
    time.sleep(3)
    if not box.is_running("agentic-both"):
        log = box.session.logs(
            "agentic-both", tail=80, log="/workspace/agentic_both.log"
        )
        raise RuntimeError(f"agentic-both died immediately:\n{log}")
    print(
        f"""
LAUNCHED run={run} instance={box.instance_id}
  tmux: agentic-both
  log:  /workspace/agentic_both.log

Monitor:
  cd ~/workspace/gpubox && PYTHONUNBUFFERED=1 .venv/bin/python -u - <<'PY'
  from gpubox.api import Box
  b = Box.attach({run!r})
  print('running', b.is_running('agentic-both') if b else None)
  print(b.session.logs('agentic-both', tail=40, log='/workspace/agentic_both.log') if b else 'gone')
  PY

Pull when ALL_DONE:
  b = Box.attach({run!r})
  b.pull(".../out", ".../out/vast_agentic")
  b.destroy()
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
