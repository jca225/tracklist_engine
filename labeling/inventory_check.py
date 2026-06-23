"""Pre-pull inventory gate — evaluate slot satisfaction against pi-storage.

Read-only over SSH. Used by pull_set_for_alignment --check and make check-inventory.
"""

from __future__ import annotations

from typing import Any
from core.slot_inventory import (
    SlotSatisfaction,
    evaluate_slot,
    format_satisfaction_report,
    is_blocking,
    is_warning,
    slot_claim_from_row,
)


def fetch_slot_rows(set_id: str, ssh_sqlite) -> list[dict[str, Any]]:
    """Load set_track_slots rows for inventory check."""
    try:
        rows = ssh_sqlite(f"""
            SELECT set_id, row_index, slot_label,
                   COALESCE(recording_id, track_id) AS recording_id,
                   track_id, full_name, claimed_stem, claimed_variant,
                   is_concurrent, layer_role, constituents_json
            FROM set_track_slots
            WHERE set_id = '{set_id}'
            ORDER BY row_index;
        """)
    except Exception:
        rows = ssh_sqlite(f"""
            SELECT set_id, row_index, slot_label,
                   COALESCE(recording_id, track_id) AS recording_id,
                   track_id, full_name, claimed_stem, claimed_variant,
                   is_concurrent
            FROM set_track_slots
            WHERE set_id = '{set_id}'
            ORDER BY row_index;
        """)
    return [r for r in rows if r.get("slot_label") and r.get("recording_id")]


def fetch_candidates_for_set(
    set_id: str, ssh_sqlite
) -> dict[str, list[dict[str, Any]]]:
    rows = ssh_sqlite(f"""
        WITH wanted(tid) AS (
            SELECT COALESCE(recording_id, track_id)
            FROM set_track_slots
            WHERE set_id = '{set_id}'
            UNION
            SELECT track_id FROM dj_set_track_media_links
            WHERE set_id = '{set_id}'
        )
        SELECT
            ta.track_id, ta.recording_id, ta.track_audio_id,
            ta.path, ta.stem, ta.variant, ta.platform, ta.is_reference,
            ta.downloaded_at
        FROM track_audio ta
        WHERE ta.track_id IN (SELECT tid FROM wanted)
           OR ta.recording_id IN (SELECT tid FROM wanted);
    """)
    by_tid: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        for key in (r.get("track_id"), r.get("recording_id")):
            if key:
                by_tid.setdefault(str(key), []).append(r)
    return by_tid


def evaluate_set_inventory(
    set_id: str,
    ssh_sqlite,
) -> list[SlotSatisfaction]:
    slot_rows = fetch_slot_rows(set_id, ssh_sqlite)
    if not slot_rows:
        return []
    candidates_by_tid = fetch_candidates_for_set(set_id, ssh_sqlite)
    results: list[SlotSatisfaction] = []
    for row in slot_rows:
        claim = slot_claim_from_row(row)
        cands = candidates_by_tid.get(claim.recording_id, [])
        results.append(evaluate_slot(claim, cands))
    return results


def run_inventory_check(
    set_id: str,
    ssh_sqlite,
    *,
    warn_only: bool = False,
    discord_root: Path | None = None,
) -> int:
    """Print report; return exit code (0 ok, 1 blocking issues)."""
    results = evaluate_set_inventory(set_id, ssh_sqlite)
    if not results:
        print(f"No materialized slots for set {set_id}")
        return 1

    discord_hints = _discord_hints(discord_root) if discord_root else {}

    n_block = sum(1 for r in results if is_blocking(r))
    n_warn = sum(1 for r in results if is_warning(r))
    n_ok = sum(1 for r in results if not is_blocking(r) and not is_warning(r))

    print(f"Inventory check: {set_id}")
    print(f"  satisfied: {n_ok}  fallback(warn): {n_warn}  blocking: {n_block}")
    print(format_satisfaction_report(results))

    if discord_hints:
        print("\nDiscord index hints (missing payload/acapella):")
        for r in results:
            if not is_blocking(r):
                continue
            key = _norm_discord_key(r.claim.display_name, r.claim.claimed_stem)
            paths = discord_hints.get(key, [])
            for p in paths[:3]:
                print(f"  {r.claim.slot_label}: {p}")

    if n_block and not warn_only:
        return 1
    return 0


def _norm_discord_key(display_name: str, stem: str) -> str:
    return f"{display_name.lower()}|{stem}"


def _discord_hints(root: Path | None) -> dict[str, list[str]]:
    """Index discord_scrape manifest by normalized artist-title + stem."""
    if root is None:
        root = Path.home() / "discord_stems"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return {}
    import json

    try:
        assets = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, list[str]] = {}
    for a in assets.values() if isinstance(assets, dict) else []:
        if not isinstance(a, dict):
            continue
        fname = a.get("local_path") or a.get("filename") or ""
        channel = (a.get("channel_label") or "").lower()
        stem = (
            "acappella"
            if "acap" in channel
            else "instrumental"
            if "instr" in channel
            else "regular"
        )
        title = (a.get("title") or fname).lower()
        key = f"{title}|{stem}"
        out.setdefault(key, []).append(str(root / fname) if fname else fname)
    return out


def satisfaction_to_manifest_fields(s: SlotSatisfaction) -> dict[str, Any]:
    """JSON-serializable manifest extras for a track entry."""
    out: dict[str, Any] = {
        "layer_role": s.claim.layer_role,
        "satisfaction": s.status.value,
        "gap": s.detail,
        "bed_slot": s.claim.bed_slot,
        "constituent_ids": list(s.claim.constituent_ids),
    }
    if s.resolve_tier is not None:
        out["resolve_tier"] = s.resolve_tier
    return out
