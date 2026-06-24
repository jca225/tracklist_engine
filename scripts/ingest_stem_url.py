#!/usr/bin/env python3
"""Mac driver: ingest a stem (acappella/instrumental) on pi-storage from a URL or file.

Downloads on pi via yt-dlp into canonical objects/ (no Mac Downloads hop).
Wraps acquire_variant (add) or replace_stem_audio (replace) over SSH.

See docs/stem_discovery_playbook.md.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PI_HOST = "pi-storage"
PI_REPO = "~/tracklist_engine"
PI_PYTHON = "venvs/audio/bin/python"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
MIN_PI_SHA = "507b08d"  # stem replace tooling baseline

_IDENTITY_VERDICT_RE = re.compile(
    r"identity-check \[([A-Z_]+)\]:",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube / YT Music URL (downloaded on pi)")
    src.add_argument(
        "--file", type=Path, help="Local audio file (scp to pi, then ingest)"
    )

    p.add_argument(
        "--track-audio-id",
        type=int,
        default=None,
        help="Replace this track_audio row (replace mode)",
    )
    p.add_argument(
        "--track-id",
        default=None,
        help="track_id for add mode (required with --role if no --track-audio-id)",
    )
    p.add_argument(
        "--role",
        choices=("acappella", "instrumental", "acapella", "vocals", "instr"),
        default=None,
        help="Stem role for add mode",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="Add mode only: set is_reference=1 (default add: no promote)",
    )
    p.add_argument("--set-id", default=None)
    p.add_argument("--position", default=None)
    p.add_argument(
        "--reason",
        required=True,
        help="Correction ledger note, e.g. quality:good|identity:OK|...",
    )
    p.add_argument(
        "--player-id",
        default=None,
        help="player_id for --file ingest (defaults to filename stem)",
    )
    p.add_argument(
        "--pull",
        action="store_true",
        help="After ingest, run pull_set_for_alignment locally for --set-id",
    )
    p.add_argument(
        "--aligning-dest",
        default="~/aligning",
        help="Destination root for --pull (default: ~/aligning)",
    )
    p.add_argument(
        "--fail-on",
        default="",
        help="Comma-separated verdicts that exit 1: fallback, wrong_song, "
        "duration_mismatch (never blocks acappella WEAK_SIGNAL)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print remote command only")
    p.add_argument("--no-log", action="store_true", help="Skip correction ledger on pi")
    p.add_argument(
        "--no-case-log",
        action="store_true",
        help="Skip the local acquisition-case attempt log (needs --set-id)",
    )
    p.add_argument(
        "--case-root",
        default="data/acquisition_cases",
        help="Root dir for acquisition-case JSONL (default: data/acquisition_cases)",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pi git SHA and materialize warnings",
    )
    return p.parse_args(argv)


def _norm_role(role: str) -> str:
    r = role.strip().lower()
    if r in ("acappella", "acapella", "vocals"):
        return "acappella"
    if r in ("instrumental", "instr"):
        return "instrumental"
    sys.exit(f"unknown role {role!r}")


def _remote_shell(parts: list[str]) -> str:
    """Build `cd repo && venvs/audio/bin/python ...` for ssh."""
    inner = " ".join(shlex.quote(p) for p in parts)
    return f"cd {PI_REPO} && {PI_PYTHON} {inner}"


def build_remote_command(
    args: argparse.Namespace, *, remote_file: str | None = None
) -> list[str]:
    """Argv fragment after python executable (script path + flags)."""
    if args.track_audio_id is not None:
        cmd = [
            "scripts/replace_stem_audio.py",
            "--track-audio-id",
            str(args.track_audio_id),
        ]
        if remote_file:
            cmd.extend(["--file", remote_file])
            if args.player_id:
                cmd.extend(["--player-id", args.player_id])
        elif args.url:
            cmd.extend(["--url", args.url])
        else:
            sys.exit("internal: replace mode needs url or file")
        # replace_stem_audio promotes by default
    else:
        if not args.track_id or not args.role:
            sys.exit(
                "add mode needs --track-id and --role (or use --track-audio-id to replace)"
            )
        role = _norm_role(args.role)
        cmd = ["scripts/acquire_variant.py"]
        if args.url:
            cmd.append(args.url)
        cmd.extend(["--role", role, "--track-id", args.track_id])
        if remote_file:
            cmd.extend(["--file", remote_file])
            cmd.extend(["--player-id", args.player_id or Path(remote_file).stem])
        elif not args.url:
            sys.exit("internal: add mode needs url or file")
        if not args.promote:
            cmd.append("--no-promote-reference")

    cmd.extend(["--reason", args.reason])
    if args.set_id:
        cmd.extend(["--set-id", args.set_id])
    if args.position:
        if args.track_audio_id is not None:
            cmd.extend(["--position", str(args.position)])
        elif str(args.position).isdigit():
            cmd.extend(["--slot", str(args.position)])
        # w-slots (013w1): acquire_variant ledger uses set_id + reason only
    if args.no_log:
        cmd.append("--no-log")
    return cmd


def _pi_git_sha() -> str | None:
    try:
        r = subprocess.run(
            ["ssh", PI_HOST, f"cd {PI_REPO} && git rev-parse --short HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _materialize_warning() -> str | None:
    try:
        r = subprocess.run(
            [
                "ssh",
                PI_HOST,
                f"sqlite3 {CANONICAL_DB} 'SELECT COUNT(*) FROM track_metadata'",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return "could not query track_metadata on pi-storage"
    try:
        n = int(r.stdout.strip())
    except ValueError:
        return None
    if n == 0:
        return (
            "track_metadata is empty — tokenizer.materialize may still be running; "
            "ingest is OK but set_track_slots audits may be incomplete"
        )
    return None


def _preflight(args: argparse.Namespace) -> None:
    if args.skip_preflight:
        return
    sha = _pi_git_sha()
    if sha is None:
        print("warning: could not read pi-storage git SHA", file=sys.stderr)
    elif sha < MIN_PI_SHA:
        print(
            f"warning: pi-storage @ {sha} is older than {MIN_PI_SHA} — "
            f"run: ssh {PI_HOST} 'cd {PI_REPO} && git pull'",
            file=sys.stderr,
        )
    warn = _materialize_warning()
    if warn:
        print(f"warning: {warn}", file=sys.stderr)


def _scp_to_pi(local: Path) -> str:
    remote = f"/tmp/ingest_stem_{local.name}"
    subprocess.run(
        ["scp", str(local), f"{PI_HOST}:{remote}"],
        check=True,
    )
    return remote


def _parse_verdict(stdout: str) -> str | None:
    hits = _IDENTITY_VERDICT_RE.findall(stdout)
    return hits[-1] if hits else None


def _fail_on_set(spec: str) -> frozenset[str]:
    if not spec.strip():
        return frozenset()
    mapping = {
        "fallback": "FALLBACK_TO_ORIGINAL",
        "wrong_song": "WRONG_SONG",
        "duration_mismatch": "DURATION_MISMATCH",
        "duration": "DURATION_MISMATCH",
    }
    out: set[str] = set()
    for part in spec.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in mapping:
            sys.exit(
                f"unknown --fail-on value {part!r}; use: fallback, wrong_song, duration_mismatch"
            )
        out.add(mapping[key])
    return frozenset(out)


def _resolve_track_id_on_pi(
    track_id: str | None, track_audio_id: int | None
) -> str | None:
    if track_id:
        return track_id
    if track_audio_id is None:
        return None
    sql = f"SELECT track_id FROM track_audio WHERE track_audio_id = {track_audio_id}"
    try:
        r = subprocess.run(
            ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB} {shlex.quote(sql)}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _resolve_slot_on_pi(set_id: str, recording_id: str) -> str | None:
    """The unique ``set_track_slots.slot_label`` this recording fills in the set.

    Best-effort: returns None if the recording fills no slot, fills *more than
    one* (ambiguous — don't guess a key), or the query fails. Lets batch callers
    (ingest_candidate_winners, apply_stem_matches) that pass --set-id + --track-id
    but no --position still get a correctly-keyed acquisition case, without each
    reimplementing the lookup.
    """
    sql = (
        "SELECT slot_label FROM set_track_slots "
        f"WHERE set_id = '{set_id}' AND recording_id = '{recording_id}'"
    )
    try:
        r = subprocess.run(
            ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB} {shlex.quote(sql)}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    labels = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return labels[0] if len(labels) == 1 else None


def _post_flight(
    args: argparse.Namespace, *, track_id: str | None, stem: str | None
) -> None:
    track_id = _resolve_track_id_on_pi(track_id, args.track_audio_id)
    if not track_id:
        return
    stem_clause = f" AND stem = '{stem}'" if stem else ""
    sql = (
        "SELECT track_audio_id, stem, is_reference, platform, player_id, "
        f"substr(path, -70) FROM track_audio WHERE recording_id = '{track_id}'"
        f"{stem_clause} ORDER BY track_audio_id DESC LIMIT 3"
    )
    try:
        r = subprocess.run(
            [
                "ssh",
                PI_HOST,
                f"sqlite3 -header -column {CANONICAL_DB} {shlex.quote(sql)}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"post-flight query failed: {e}", file=sys.stderr)
        return
    if r.stdout.strip():
        print("\npost-flight track_audio (latest):")
        print(r.stdout.strip())
    if args.set_id:
        csql = (
            "SELECT correction_id, action, stem_value, substr(reason,1,50) "
            f"FROM track_audio_correction WHERE track_id = '{track_id}' "
            "ORDER BY correction_id DESC LIMIT 1"
        )
        cr = subprocess.run(
            [
                "ssh",
                PI_HOST,
                f"sqlite3 -header -column {CANONICAL_DB} {shlex.quote(csql)}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cr.stdout.strip():
            print("\nlatest correction:")
            print(cr.stdout.strip())


def _log_case_attempt(
    args: argparse.Namespace, *, recording_id: str | None, verdict: str | None
) -> None:
    """Append an Attempt to the local acquisition case for this fix.

    No-op unless ``--set-id`` is given (the case is keyed per set). A replace
    promotes by default; an add promotes only with ``--promote``.
    """
    if args.no_case_log or not args.set_id:
        return
    if not recording_id:
        print("case-log: could not resolve recording_id; skipping", file=sys.stderr)
        return
    # Slot keys the case. Prefer the explicit --position; otherwise resolve the
    # unique slot this recording fills in the set (covers batch ingests that pass
    # --set-id + --track-id but no --position).
    position = args.position or _resolve_slot_on_pi(args.set_id, recording_id)
    if not position:
        print(
            "case-log: no unique slot_label for (set, recording); skipping",
            file=sys.stderr,
        )
        return

    from core.acquisition_case import (
        Actor,
        Attempt,
        AttemptAction,
        AttemptVerdict,
        CaseClaim,
        ProblemClass,
        Resolution,
        record_attempt,
    )
    from core.identity import normalize_stem

    stem = _norm_role(args.role) if args.role else "regular"
    promoted = args.track_audio_id is not None or args.promote
    source = args.url or (f"file://{args.file}" if args.file else None)
    checks = {"identity": verdict} if verdict else {}

    attempt = Attempt(
        action=AttemptAction.INGEST_URL,
        actor=Actor.HUMAN,
        url=source,
        platform="online_candidate",
        verdict=AttemptVerdict.PROMOTE if promoted else AttemptVerdict.ACCEPT,
        checks=checks,
        notes=args.reason,
    )
    resolution = (
        Resolution(
            track_audio_id=args.track_audio_id,
            ref_source="online_candidate",
            source_path=source,
        )
        if promoted
        else None
    )
    add_problems = (
        (ProblemClass.SUBOPTIMAL_STEM,) if stem in ("acappella", "instrumental") else ()
    )
    case = record_attempt(
        set_id=args.set_id,
        slot_label=str(position),
        recording_id=recording_id,
        attempt=attempt,
        claim=CaseClaim(recording_id=recording_id, stem=normalize_stem(stem)),
        resolution=resolution,
        add_problems=add_problems,
        root=args.case_root,
    )
    print(f"\ncase-log: appended attempt to {case.case_id} ({case.status.value})")


def _print_ytdlp_help() -> None:
    print(
        "\npi-storage yt-dlp may need recovery — see .claude/skills/audio-pipeline-debug/SKILL.md\n"
        "Mac fallback:\n"
        "  venvs/audio/bin/yt-dlp -f 'bestaudio[ext=m4a]/bestaudio' -o /tmp/stem.m4a 'URL'\n"
        "  venvs/audio/bin/python scripts/ingest_stem_url.py --file /tmp/stem.m4a "
        "--track-id ... --role acappella --reason '...'\n",
        file=sys.stderr,
    )


def _run_pull(args: argparse.Namespace) -> int:
    if not args.set_id:
        print("--pull requires --set-id", file=sys.stderr)
        return 2
    pull = REPO / "labeling" / "pull_set_for_alignment.py"
    cmd = [
        sys.executable,
        str(pull),
        args.set_id,
        "--dest",
        args.aligning_dest,
    ]
    print(f"\npull: {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _preflight(args)

    remote_file: str | None = None
    if args.file:
        if not args.file.is_file():
            print(f"file not found: {args.file}", file=sys.stderr)
            return 2
        if not args.dry_run:
            print(f"scp {args.file} -> {PI_HOST}:/tmp/...")
            remote_file = _scp_to_pi(args.file.resolve())

    remote_argv = build_remote_command(args, remote_file=remote_file)
    shell = _remote_shell(remote_argv)
    ssh_cmd = ["ssh", PI_HOST, shell]

    if args.dry_run:
        print("dry-run remote command:")
        print(f"  {' '.join(shlex.quote(c) for c in ssh_cmd)}")
        return 0

    print(f"remote: {shell}")
    r = subprocess.run(ssh_cmd, capture_output=True, text=True, errors="replace")
    if r.stdout:
        print(r.stdout, end="" if r.stdout.endswith("\n") else "\n")
    if r.stderr:
        print(r.stderr, end="" if r.stderr.endswith("\n") else "\n", file=sys.stderr)

    if r.returncode != 0:
        _print_ytdlp_help()
        return r.returncode

    verdict = _parse_verdict(r.stdout + r.stderr)
    if verdict:
        print(f"\nidentity verdict: {verdict}")
    blocked = _fail_on_set(args.fail_on)
    if verdict and verdict in blocked:
        print(f"aborting (--fail-on includes {verdict})", file=sys.stderr)
        return 1

    stem = _norm_role(args.role) if args.role else None
    _post_flight(args, track_id=args.track_id, stem=stem)

    recording_id = _resolve_track_id_on_pi(args.track_id, args.track_audio_id)
    _log_case_attempt(args, recording_id=recording_id, verdict=verdict)

    if args.pull:
        return _run_pull(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
