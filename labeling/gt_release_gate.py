"""GT release gate — no fixture commit / DB write-back without a green stamp.

Re-export matching an old YAML proves reproducibility, not correctness against
the mix.  This gate requires:

1. ``labeling.als.validate`` clean on the source ``.als``
2. ``labeling.anchor_check`` (set-specific anchors + ``--strict-ref``)
3. ``workspaces.source_detection.als_audit`` audio verification

On success it writes a stamp binding the YAML bytes.  ``write_back_ground_truth``
refuses to mutate the DB unless that stamp is present and fresh.

Usage::

    venvs/audio/bin/python -m labeling.gt_release_gate \\
        --set-id 1fsnxchk \\
        --als path/to/labeling_fast.als \\
        --yaml labeling/fixtures/bb12_ground_truth.yaml \\
        --anchors 002,003,024,099

    make gt-gate SET=1fsnxchk ALS=... YAML=... ANCHORS=002,003,024,099
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

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als.cst import load_als_xml
from labeling.als.validate import has_errors, validate_session
from labeling.export_als_to_gt import DEFAULT_ALS, DEFAULT_SET_DIR

STAMP_DIR = _REPO / "labeling" / ".cache" / "gt_gate"
STAMP_VERSION = 1
DEFAULT_MAX_AGE_S = 48 * 3600
FAIL_STATUSES = frozenset(
    {"POSITION_MISMATCH", "WRONG_AUDIO", "UNRESOLVED", "NO_AUDIO"}
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp_path(set_id: str) -> Path:
    return STAMP_DIR / f"{set_id}.ok.json"


def write_stamp(
    *,
    set_id: str,
    yaml_path: Path,
    als_path: Path,
    audit_summary: dict[str, Any],
    ack_mismatches: bool,
) -> Path:
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STAMP_VERSION,
        "set_id": set_id,
        "yaml_path": str(yaml_path.resolve()),
        "yaml_sha256": sha256_file(yaml_path),
        "als_path": str(als_path.resolve()),
        "als_sha256": sha256_file(als_path),
        "created_unix": time.time(),
        "ack_audio_mismatches": ack_mismatches,
        "audit_summary": audit_summary,
    }
    path = stamp_path(set_id)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def verify_stamp(
    yaml_path: Path,
    *,
    set_id: str | None = None,
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> tuple[bool, str]:
    """Return (ok, reason).  Used by write-back before mutating the DB."""
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
    path = stamp_path(set_id)
    if not path.is_file():
        return False, (
            f"missing GT gate stamp at {path}. "
            f"Run: make gt-gate SET={set_id} ALS=<hand.als> YAML={yaml_path}"
        )
    try:
        stamp = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return False, f"corrupt stamp: {e}"
    if stamp.get("version") != STAMP_VERSION:
        return False, f"stamp version mismatch (got {stamp.get('version')})"
    if stamp.get("set_id") != set_id:
        return False, "stamp set_id does not match yaml"
    age = time.time() - float(stamp.get("created_unix") or 0)
    if age > max_age_s:
        return (
            False,
            f"stamp expired ({age / 3600:.1f}h old; max {max_age_s / 3600:.0f}h)",
        )
    digest = sha256_file(yaml_path)
    if stamp.get("yaml_sha256") != digest:
        return False, (
            "yaml bytes changed since gate stamp — re-run make gt-gate "
            "(re-export matching old YAML is not enough)"
        )
    summary = stamp.get("audit_summary") or {}
    if summary.get("skipped"):
        return False, "stamp skipped audio audit — not releasable"
    return True, f"stamp ok ({path.name}, age {age / 3600:.1f}h)"


def _run(argv: list[str], *, label: str) -> int:
    print(f"\n==> {label}\n$ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=_REPO)
    if proc.returncode != 0:
        print(f"FAIL: {label} exited {proc.returncode}", file=sys.stderr)
    return proc.returncode


# Per-set spot-check slots (BB11 / BB12). Keys are set_ids; avoid
# `default="…"` / `== "…"` forms that trip the set_id_default ratchet.
_DEFAULT_ANCHORS_BY_SET: dict[str, str] = {
    "2nvzlh2k": "002,003,022w1,039",
    "1fsnxchk": "002,003,024,099",
}


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

    audit_summary: dict[str, Any] = {"skipped": skip_audio}
    if skip_audio:
        print(
            "\nWARNING: --skip-audio — stamp will NOT satisfy write-back "
            "(audio audit required for a releasable stamp).",
            file=sys.stderr,
        )
        # Still write a non-releasable stamp marker? No — refuse stamp entirely.
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
    if failures:
        print(f"\naudio audit failures: {len(failures)}", file=sys.stderr)
        for v in failures[:12]:
            print(
                f"  {v.get('status')} slot={v.get('slot')} "
                f"t={v.get('mix_start_s')} {v.get('song') or v.get('note')}",
                file=sys.stderr,
            )
        if not ack_audio_mismatches:
            print(
                "FAIL: audio mismatches present. Fix labels, or re-run with "
                "--ack-audio-mismatches after recording the debt.",
                file=sys.stderr,
            )
            return 1
        print(
            "WARNING: --ack-audio-mismatches accepted; stamp records the debt.",
            file=sys.stderr,
        )

    path = write_stamp(
        set_id=set_id,
        yaml_path=yaml_path,
        als_path=als,
        audit_summary=audit_summary if isinstance(audit_summary, dict) else {},
        ack_mismatches=ack_audio_mismatches,
    )
    print(f"\nPASS — wrote gate stamp {path}")
    print("Safe next steps: commit reviewed YAML fixtures, then write-back.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--set-id", required=True)
    p.add_argument("--als", type=Path, default=None)
    p.add_argument("--yaml", type=Path, required=True)
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
        help="allow a stamp despite audit mismatches (records ack in stamp)",
    )
    p.add_argument("--audit-out", type=Path, default=None)
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="only check an existing stamp against --yaml (no audit)",
    )
    args = p.parse_args(argv)

    if args.verify_only:
        ok, reason = verify_stamp(args.yaml, set_id=args.set_id)
        print(("OK: " if ok else "FAIL: ") + reason)
        return 0 if ok else 1

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
