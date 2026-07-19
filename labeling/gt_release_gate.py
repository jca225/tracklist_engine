"""GT release gate — no fixture commit / DB write-back without a green stamp.

Re-export matching an old YAML proves reproducibility, not correctness against
the mix.  XML codec round-trip is **not** enough (it missed distant-clip merges).
This gate requires:

1. ``labeling.als.validate`` clean on the source ``.als``
2. ``labeling.anchor_check`` (set-specific anchors + ``--strict-ref``)
3. ``workspaces.source_detection.als_audit`` (labels vs DJ mix)
4. ``labeling.audio_roundtrip`` denotational law (arrangement ↔ GT audio)

On success it writes:

- a local stamp under ``labeling/.cache/gt_gate/`` (unlocks write-back)
- a committed stamp under ``labeling/fixtures/gt_gate_stamps/`` (unlocks
  fixture commits / status regen via pre-commit)

Known leftover mismatches live in ``labeling/fixtures/gt_audit_acks.yaml``.

Usage::

    make gt-gate SET=1fsnxchk ALS=... YAML=labeling/fixtures/bb12_ground_truth.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als.cst import load_als_xml
from labeling.als.validate import has_errors, validate_session
from labeling.export_als_to_gt import DEFAULT_ALS, DEFAULT_SET_DIR

STAMP_DIR = _REPO / "labeling" / ".cache" / "gt_gate"
FIXTURE_STAMP_DIR = _REPO / "labeling" / "fixtures" / "gt_gate_stamps"
ACKS_PATH = _REPO / "labeling" / "fixtures" / "gt_audit_acks.yaml"
STAMP_VERSION = 1
DEFAULT_MAX_AGE_S = 48 * 3600
ACK_TOL_S = 0.5
FAIL_STATUSES = frozenset(
    {"POSITION_MISMATCH", "WRONG_AUDIO", "UNRESOLVED", "NO_AUDIO"}
)

# Per-set spot-check slots (BB11 / BB12). Keys are set_ids; avoid
# `default="…"` / `== "…"` forms that trip the set_id_default ratchet.
_DEFAULT_ANCHORS_BY_SET: dict[str, str] = {
    "2nvzlh2k": "002,003,022w1,039",
    "1fsnxchk": "002,003,024,099",
}

# Fixtures that must be stamped before regenerating docs/alignment_status.md.
_STATUS_FIXTURES: tuple[Path, ...] = (
    _REPO / "labeling" / "fixtures" / "bb11_ground_truth.yaml",
    _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp_path(set_id: str) -> Path:
    return STAMP_DIR / f"{set_id}.ok.json"


def fixture_stamp_path(set_id: str) -> Path:
    return FIXTURE_STAMP_DIR / f"{set_id}.json"


def _stamp_payload(
    *,
    set_id: str,
    yaml_path: Path,
    als_path: Path,
    audit_summary: dict[str, Any],
    ack_mismatches: bool,
    acked_failures: list[dict[str, Any]],
    audio_roundtrip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": STAMP_VERSION,
        "set_id": set_id,
        "yaml_path": str(yaml_path.resolve()),
        "yaml_sha256": sha256_file(yaml_path),
        "als_path": str(als_path.resolve()),
        "als_sha256": sha256_file(als_path) if als_path.is_file() else "",
        "created_unix": time.time(),
        "ack_audio_mismatches": ack_mismatches,
        "acked_failures": acked_failures,
        "audit_summary": audit_summary,
        "audio_roundtrip": audio_roundtrip or {},
    }


def write_stamp(
    *,
    set_id: str,
    yaml_path: Path,
    als_path: Path,
    audit_summary: dict[str, Any],
    ack_mismatches: bool,
    acked_failures: list[dict[str, Any]] | None = None,
    audio_roundtrip: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write local + committed stamps. Returns (local_path, fixture_path)."""
    payload = _stamp_payload(
        set_id=set_id,
        yaml_path=yaml_path,
        als_path=als_path,
        audit_summary=audit_summary,
        ack_mismatches=ack_mismatches,
        acked_failures=acked_failures or [],
        audio_roundtrip=audio_roundtrip,
    )
    text = json.dumps(payload, indent=2) + "\n"
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURE_STAMP_DIR.mkdir(parents=True, exist_ok=True)
    local = stamp_path(set_id)
    committed = fixture_stamp_path(set_id)
    local.write_text(text)
    committed.write_text(text)
    return local, committed


def load_acks(path: Path = ACKS_PATH) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    sets = raw.get("sets") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for set_id, body in sets.items():
        entries = (body or {}).get("entries") or []
        out[str(set_id)] = [e for e in entries if isinstance(e, dict)]
    return out


def _failure_key(v: dict[str, Any]) -> tuple[str, str, float]:
    return (
        str(v.get("status") or ""),
        str(v.get("slot") or ""),
        float(v.get("mix_start_s") or 0.0),
    )


def partition_failures(
    failures: list[dict[str, Any]],
    acks: list[dict[str, Any]],
    *,
    tol_s: float = ACK_TOL_S,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split audit failures into (unacked, acked)."""
    remaining = list(acks)
    unacked: list[dict[str, Any]] = []
    acked: list[dict[str, Any]] = []
    for fail in failures:
        status, slot, t = _failure_key(fail)
        match_i: int | None = None
        for i, ack in enumerate(remaining):
            if str(ack.get("status") or "") != status:
                continue
            if str(ack.get("slot") or "") != slot:
                continue
            if abs(float(ack.get("mix_start_s") or 0.0) - t) > tol_s:
                continue
            match_i = i
            break
        if match_i is None:
            unacked.append(fail)
        else:
            acked.append(fail)
            remaining.pop(match_i)
    return unacked, acked


def _read_stamp(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def verify_stamp(
    yaml_path: Path,
    *,
    set_id: str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
    allow_committed: bool = True,
) -> tuple[bool, str]:
    """Return (ok, reason). Used by write-back before mutating the DB."""
    if set_id is None:
        from labeling.ground_truth.schema import load as load_gt
        from core.result import Err, Ok

        match load_gt(yaml_path):
            case Err(e):
                return False, f"yaml load failed: {e.detail}"
            case Ok(gt):
                set_id = gt.set_id
    if not set_id:
        return False, "yaml has no set_id"

    digest = sha256_file(yaml_path)
    candidates: list[tuple[str, Path]] = [("local", stamp_path(set_id))]
    if allow_committed:
        candidates.append(("committed", fixture_stamp_path(set_id)))

    last_reason = "missing GT gate stamp"
    for kind, path in candidates:
        stamp = _read_stamp(path)
        if stamp is None:
            continue
        if stamp.get("version") != STAMP_VERSION:
            last_reason = f"{kind} stamp version mismatch"
            continue
        if stamp.get("set_id") != set_id:
            last_reason = f"{kind} stamp set_id mismatch"
            continue
        if stamp.get("yaml_sha256") != digest:
            last_reason = (
                f"{kind} stamp yaml sha mismatch — re-run make gt-gate "
                "(re-export matching old YAML is not enough)"
            )
            continue
        summary = stamp.get("audit_summary") or {}
        if summary.get("skipped"):
            last_reason = f"{kind} stamp skipped audio audit"
            continue
        rt = stamp.get("audio_roundtrip") or {}
        if not rt.get("ok"):
            last_reason = (
                f"{kind} stamp missing denotational audio_roundtrip — "
                "re-run make gt-gate (XML round-trip is not enough)"
            )
            continue
        age = time.time() - float(stamp.get("created_unix") or 0)
        if age > max_age_s:
            last_reason = (
                f"{kind} stamp expired ({age / 3600:.1f}h old; "
                f"max {max_age_s / 3600:.0f}h)"
            )
            continue
        return True, f"{kind} stamp ok ({path.name}, age {age / 3600:.1f}h)"

    return False, (
        f"{last_reason}. Run: make gt-gate SET={set_id} ALS=<hand.als> YAML={yaml_path}"
    )


def verify_fixture_stamp(yaml_path: Path) -> tuple[bool, str]:
    """Committed stamp must match YAML bytes (no age — for pre-commit / CI)."""
    from labeling.ground_truth.schema import load as load_gt
    from core.result import Err, Ok

    match load_gt(yaml_path):
        case Err(e):
            return False, f"yaml load failed: {e.detail}"
        case Ok(gt):
            set_id = gt.set_id
    path = fixture_stamp_path(set_id)
    stamp = _read_stamp(path)
    if stamp is None:
        return False, (
            f"missing committed stamp {path}. Run make gt-gate and stage "
            f"labeling/fixtures/gt_gate_stamps/{set_id}.json"
        )
    if stamp.get("yaml_sha256") != sha256_file(yaml_path):
        return False, (
            f"committed stamp out of date for {yaml_path.name} — re-run "
            f"make gt-gate and stage the updated stamp"
        )
    return True, f"committed stamp ok for {set_id}"


def check_fixture_paths(paths: list[Path]) -> int:
    rc = 0
    for path in paths:
        ok, reason = verify_fixture_stamp(path)
        print(("OK: " if ok else "FAIL: ") + reason)
        if not ok:
            rc = 1
    return rc


def check_status_regen() -> int:
    """Fixtures feeding docs/alignment_status.md must have matching stamps."""
    print("==> gt-gate preflight for alignment_status regen")
    return check_fixture_paths(list(_STATUS_FIXTURES))


def _run(argv: list[str], *, label: str) -> int:
    print(f"\n==> {label}\n$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=_REPO)
    if proc.returncode != 0:
        print(f"FAIL: {label} exited {proc.returncode}", file=sys.stderr)
    return proc.returncode


def _default_anchors(set_id: str) -> str:
    return _DEFAULT_ANCHORS_BY_SET.get(set_id, "002,003")


def _audit_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        v
        for v in (report.get("verdicts") or [])
        if str(v.get("status") or "") in FAIL_STATUSES
    ]


def run_gate(
    *,
    set_id: str,
    als: Path,
    yaml_path: Path,
    set_dir: Path | None,
    anchors: str,
    skip_audio: bool,
    ack_audio_mismatches: bool,
    audit_out: Path | None,
) -> int:
    if not als.is_file():
        print(f"als not found: {als}", file=sys.stderr)
        return 2
    if not yaml_path.is_file():
        print(f"yaml not found: {yaml_path}", file=sys.stderr)
        return 2

    print("==> validate .als", flush=True)
    diags = validate_session(load_als_xml(als))
    if has_errors(diags):
        for d in diags:
            print(f"  {d.severity}: {d.message}", file=sys.stderr)
        print("FAIL: als validate", file=sys.stderr)
        return 1
    print("OK — als validate clean")

    py = str(_REPO / "venvs" / "audio" / "bin" / "python")
    if not Path(py).is_file():
        py = sys.executable
    anchor_argv = [
        py,
        "-m",
        "labeling.anchor_check",
        "--yaml",
        str(yaml_path),
        "--als",
        str(als),
        "--anchors",
        anchors,
        "--strict-ref",
    ]
    if set_dir is not None:
        anchor_argv.extend(["--set-dir", str(set_dir)])
    if _run(anchor_argv, label="anchor_check --strict-ref") != 0:
        return 1

    if skip_audio:
        print(
            "\nWARNING: --skip-audio — stamp will NOT satisfy write-back "
            "(audio audit required for a releasable stamp).",
            file=sys.stderr,
        )
        print("FAIL: refuse to stamp without audio audit", file=sys.stderr)
        return 1

    out_path = audit_out or (STAMP_DIR / f"{set_id}_als_audit.json")
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    audit_argv = [
        py,
        "-m",
        "workspaces.source_detection.als_audit",
        "--set-id",
        set_id,
        "--als",
        str(als),
        "--out",
        str(out_path),
    ]
    if _run(audit_argv, label="als_audit (audio)") != 0:
        return 1
    report = json.loads(out_path.read_text())
    audit_summary = report.get("summary") or {}
    failures = _audit_failures(report)
    acks = load_acks().get(set_id, [])
    unacked, acked = partition_failures(failures, acks)

    if acked:
        print(f"\nacked via gt_audit_acks.yaml: {len(acked)}")
        for v in acked[:12]:
            print(
                f"  ACK {v.get('status')} slot={v.get('slot')} t={v.get('mix_start_s')}"
            )
    if unacked:
        print(f"\naudio audit failures (unacked): {len(unacked)}", file=sys.stderr)
        for v in unacked[:12]:
            print(
                f"  {v.get('status')} slot={v.get('slot')} "
                f"t={v.get('mix_start_s')} {v.get('song') or v.get('note')}",
                file=sys.stderr,
            )
        if not ack_audio_mismatches:
            print(
                "FAIL: unacked audio mismatches. Fix labels, add entries to "
                f"{ACKS_PATH.relative_to(_REPO)}, or pass "
                "--ack-audio-mismatches (blanket; prefer the ack file).",
                file=sys.stderr,
            )
            return 1
        print(
            "WARNING: --ack-audio-mismatches blanket-accepted unacked failures; "
            "record them in gt_audit_acks.yaml.",
            file=sys.stderr,
        )

    if set_dir is None or not set_dir.is_dir():
        print(
            "FAIL: --set-dir required for denotational audio round-trip",
            file=sys.stderr,
        )
        return 1
    print("\n==> audio round-trip (ALS arrangement ↔ GT denotation)", flush=True)
    from labeling.audio_roundtrip import run_roundtrip

    rt = run_roundtrip(als, set_dir, yaml_path=yaml_path)
    print(("OK: " if rt.ok else "FAIL: ") + rt.detail)
    if not rt.ok:
        print(
            "FAIL: denotational audio round-trip — GT does not recreate the "
            "arrangement audio (XML/codec round-trip is not enough).",
            file=sys.stderr,
        )
        return 1

    local, committed = write_stamp(
        set_id=set_id,
        yaml_path=yaml_path,
        als_path=als,
        audit_summary=audit_summary if isinstance(audit_summary, dict) else {},
        ack_mismatches=ack_audio_mismatches or bool(acked),
        acked_failures=[
            {
                "status": v.get("status"),
                "slot": v.get("slot"),
                "mix_start_s": v.get("mix_start_s"),
            }
            for v in acked
        ],
        audio_roundtrip=rt.as_dict(),
    )
    print(f"\nPASS — wrote local stamp {local}")
    print(f"PASS — wrote committed stamp {committed} (stage this with the YAML)")
    print("Safe next steps: commit reviewed YAML + stamp, then write-back.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--set-id", default=None)
    p.add_argument("--als", type=Path, default=None)
    p.add_argument("--yaml", type=Path, default=None)
    p.add_argument("--set-dir", type=Path, default=None)
    p.add_argument(
        "--anchors",
        default=None,
        help="Comma-separated slots for anchor_check (defaults per BB11/BB12)",
    )
    p.add_argument(
        "--skip-audio",
        action="store_true",
        help="plumbing only — refuses to stamp (cannot unlock write-back)",
    )
    p.add_argument(
        "--ack-audio-mismatches",
        action="store_true",
        help="blanket-allow unacked audit mismatches (prefer gt_audit_acks.yaml)",
    )
    p.add_argument("--audit-out", type=Path, default=None)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="only check an existing stamp against --yaml (no audit)",
    )
    p.add_argument(
        "--check-fixtures",
        nargs="+",
        type=Path,
        default=None,
        help="verify committed stamps match these YAML paths (pre-commit)",
    )
    p.add_argument(
        "--check-status-regen",
        action="store_true",
        help="verify BB11/BB12 fixture stamps before regenerating alignment_status",
    )
    args = p.parse_args(argv)

    if args.check_status_regen:
        return check_status_regen()
    if args.check_fixtures is not None:
        return check_fixture_paths(args.check_fixtures)

    if args.verify_only:
        if args.yaml is None:
            print("--verify-only requires --yaml", file=sys.stderr)
            return 2
        ok, reason = verify_stamp(args.yaml, set_id=args.set_id)
        print(("OK: " if ok else "FAIL: ") + reason)
        return 0 if ok else 1

    if not args.set_id or args.yaml is None:
        print("--set-id and --yaml are required for a full gate run", file=sys.stderr)
        return 2

    als = args.als or DEFAULT_ALS
    set_dir = args.set_dir
    if set_dir is None and DEFAULT_SET_DIR.is_dir():
        set_dir = DEFAULT_SET_DIR
    anchors = args.anchors or _default_anchors(args.set_id)
    return run_gate(
        set_id=args.set_id,
        als=als,
        yaml_path=args.yaml,
        set_dir=set_dir,
        anchors=anchors,
        skip_audio=args.skip_audio,
        ack_audio_mismatches=args.ack_audio_mismatches,
        audit_out=args.audit_out,
    )


if __name__ == "__main__":
    sys.exit(main())
