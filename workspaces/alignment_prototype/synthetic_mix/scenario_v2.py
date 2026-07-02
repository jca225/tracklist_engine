"""Sample BB12-realistic mashup windows from stem catalog."""

from __future__ import annotations

import numpy as np

from .catalog import (
    BedEntry,
    PayloadEntry,
    RegularEntry,
    StemCatalog,
    compatible,
    pitch_shift_semi,
    tempo_ratio,
)
from .sections import CurriculumV2, get_curriculum
from .timeline import (
    AcappellaSpan,
    InstrumentalBlock,
    MashupWindowV2,
    MixSlice,
    RegularSpan,
)


def _beds_compatible(a: BedEntry, b: BedEntry, cfg: CurriculumV2) -> bool:
    if a.key_pc is None or b.key_pc is None or a.bpm <= 0 or b.bpm <= 0:
        return False
    from workspaces.mashup_compat.baseline import _bpm_fold, _key_dist

    if _key_dist(a.key_pc, b.key_pc) > cfg.max_key_dist:
        return False
    if _bpm_fold(a.bpm, b.bpm) > cfg.max_bpm_fold:
        return False
    return a.recording_id != b.recording_id


def _pick_beds(
    catalog: StemCatalog, cfg: CurriculumV2, rng: np.random.Generator
) -> tuple[BedEntry, ...] | None:
    n = cfg.n_instrumentals
    if len(catalog.beds) < n:
        return None
    order = rng.permutation(len(catalog.beds))
    first = catalog.beds[order[0]]
    chosen = [first]
    for idx in order[1:]:
        if len(chosen) >= n:
            break
        bed = catalog.beds[idx]
        if all(_beds_compatible(bed, c, cfg) for c in chosen):
            chosen.append(bed)
    if len(chosen) < n:
        return None
    return tuple(chosen)


def _instr_slices(
    bed: BedEntry,
    mix_start: float,
    mix_end: float,
    cfg: CurriculumV2,
    rng: np.random.Generator,
) -> tuple[MixSlice, ...]:
    dur = mix_end - mix_start
    if dur < 12.0:
        return ()
    if rng.random() > cfg.instr_jump_prob:
        ref0 = float(rng.uniform(0.0, 20.0))
        return (
            MixSlice(
                mix_start_s=mix_start,
                mix_end_s=mix_end,
                ref_start_s=ref0,
                ref_end_s=ref0 + dur,
            ),
        )
    n_seg = int(rng.integers(cfg.instr_jump_segments[0], cfg.instr_jump_segments[1] + 1))
    cuts = sorted(rng.uniform(mix_start + 8.0, mix_end - 8.0, size=max(0, n_seg - 1)))
    bounds = [mix_start, *cuts, mix_end]
    slices: list[MixSlice] = []
    ref_cursor = float(rng.uniform(0.0, 30.0))
    for i in range(len(bounds) - 1):
        seg_dur = bounds[i + 1] - bounds[i]
        if seg_dur < 4.0:
            continue
        jump = float(rng.uniform(30.0, 120.0))
        ref_start = ref_cursor + (jump if i else 0.0)
        ref_cursor = ref_start
        slices.append(
            MixSlice(
                mix_start_s=bounds[i],
                mix_end_s=bounds[i + 1],
                ref_start_s=ref_start,
                ref_end_s=ref_start + seg_dur,
            )
        )
    return tuple(slices) if slices else (
        MixSlice(mix_start, mix_end, 0.0, dur),
    )


def _schedule_instrumentals(
    beds: tuple[BedEntry, ...],
    window_s: float,
    cfg: CurriculumV2,
    rng: np.random.Generator,
    slot_base: int,
) -> tuple[InstrumentalBlock, ...]:
    fade = cfg.handoff_crossfade_s
    n = len(beds)
    if n == 1:
        return (
            InstrumentalBlock(
                bed=beds[0],
                mix_start_s=0.0,
                mix_end_s=window_s,
                slices=_instr_slices(beds[0], 0.0, window_s, cfg, rng),
                slot_label=str(slot_base),
            ),
        )
    # Overlapping handoffs: each next bed starts before previous ends.
    seg = window_s / n
    blocks: list[InstrumentalBlock] = []
    for i, bed in enumerate(beds):
        start = max(0.0, i * seg - fade * 0.5)
        end = window_s if i == n - 1 else min(window_s, (i + 1) * seg + fade * 0.5)
        blocks.append(
            InstrumentalBlock(
                bed=bed,
                mix_start_s=start,
                mix_end_s=end,
                slices=_instr_slices(bed, start, end, cfg, rng),
                slot_label=str(slot_base + i),
            )
        )
    return tuple(blocks)


def _host_at(t: float, blocks: tuple[InstrumentalBlock, ...]) -> InstrumentalBlock:
    active = blocks[0]
    for b in blocks:
        if b.mix_start_s <= t < b.mix_end_s:
            active = b
    return active


def _pick_payloads(
    catalog: StemCatalog,
    host: BedEntry,
    cfg: CurriculumV2,
    n: int,
    rng: np.random.Generator,
) -> tuple[PayloadEntry, ...] | None:
    pool = [
        p
        for p in catalog.payloads
        if compatible(host, p, max_key_dist=cfg.max_key_dist, max_bpm_fold=cfg.max_bpm_fold)
    ]
    if len(pool) < n:
        return None
    idx = rng.choice(len(pool), size=n, replace=False)
    return tuple(pool[i] for i in idx)


def _linear_acap(
    payload: PayloadEntry,
    host: BedEntry,
    mix_start: float,
    dur: float,
    window_s: float,
    slot: str,
    parent: str,
    rng: np.random.Generator,
) -> AcappellaSpan | None:
    mix_end = min(window_s, mix_start + dur)
    if mix_end - mix_start < 10.0:
        return None
    ref_start = float(rng.uniform(10.0, 80.0))
    tr = tempo_ratio(host, payload)
    ref_dur = (mix_end - mix_start) * tr
    fade_t = mix_end - 4.0
    gain = ((mix_start, 0.0), (mix_start + 3.5, 1.0), (fade_t, 1.0), (mix_end, 0.0))
    return AcappellaSpan(
        payload=payload,
        mix_start_s=mix_start,
        mix_end_s=mix_end,
        host_bpm=host.bpm,
        ref_start_s=ref_start,
        ref_end_s=ref_start + ref_dur,
        slices=(),
        is_loop=False,
        slot_label=slot,
        parent_slot=parent,
        tempo_ratio=tr,
        pitch_shift_semi=pitch_shift_semi(host, payload),
        gain_curve=gain,
    )


def _loop_acap(
    payload: PayloadEntry,
    host: BedEntry,
    mix_start: float,
    window_s: float,
    cfg: CurriculumV2,
    slot: str,
    parent: str,
    rng: np.random.Generator,
) -> AcappellaSpan | None:
    phrase = float(rng.uniform(*cfg.loop_phrase_s))
    n_rep = int(rng.integers(cfg.loop_repeats[0], cfg.loop_repeats[1] + 1))
    total = phrase * n_rep
    mix_end = min(window_s, mix_start + total + 4.0)
    if mix_end - mix_start < phrase * 2:
        return None
    ref_lo = float(rng.uniform(20.0, 70.0))
    tr = tempo_ratio(host, payload)
    slices: list[MixSlice] = []
    t = mix_start
    for _ in range(n_rep):
        if t + phrase > mix_end:
            break
        slices.append(
            MixSlice(
                mix_start_s=t,
                mix_end_s=t + phrase,
                ref_start_s=ref_lo,
                ref_end_s=ref_lo + phrase * tr,
            )
        )
        t += phrase
    if len(slices) < 2:
        return None
    gain = ((mix_start, 0.0), (mix_start + 2.0, 1.0), (mix_end - 3.0, 1.0), (mix_end, 0.0))
    return AcappellaSpan(
        payload=payload,
        mix_start_s=mix_start,
        mix_end_s=mix_end,
        host_bpm=host.bpm,
        ref_start_s=ref_lo,
        ref_end_s=ref_lo + phrase * tr,
        slices=tuple(slices),
        is_loop=True,
        slot_label=slot,
        parent_slot=parent,
        tempo_ratio=tr,
        pitch_shift_semi=pitch_shift_semi(host, payload),
        gain_curve=gain,
    )


def _regular_span(
    regular: RegularEntry,
    host: BedEntry,
    mix_start: float,
    dur: float,
    window_s: float,
    slot: str,
    parent: str,
    rng: np.random.Generator,
) -> RegularSpan | None:
    mix_end = min(window_s, mix_start + dur)
    if mix_end - mix_start < 12.0:
        return None
    ref_start = float(rng.uniform(5.0, 60.0))
    tr = host.bpm / regular.bpm if regular.bpm > 0 else 1.0
    ref_dur = (mix_end - mix_start) * tr
    pitch = 0
    if host.key_pc is not None and regular.key_pc is not None:
        delta = (host.key_pc - regular.key_pc) % 12
        pitch = int(delta - 12 if delta > 6 else delta)
    fade_t = mix_end - 4.0
    gain = ((mix_start, 0.0), (mix_start + 3.5, 1.0), (fade_t, 1.0), (mix_end, 0.0))
    return RegularSpan(
        regular=regular,
        mix_start_s=mix_start,
        mix_end_s=mix_end,
        host_bpm=host.bpm,
        ref_start_s=ref_start,
        ref_end_s=ref_start + ref_dur,
        slot_label=slot,
        parent_slot=parent,
        tempo_ratio=tr,
        pitch_shift_semi=pitch,
        gain_curve=gain,
    )


def _regular_compatible(r: RegularEntry, host: BedEntry, cfg: CurriculumV2) -> bool:
    if r.key_pc is None or host.key_pc is None or r.bpm <= 0 or host.bpm <= 0:
        return False
    from workspaces.mashup_compat.baseline import _bpm_fold, _key_dist

    if _key_dist(r.key_pc, host.key_pc) > cfg.max_key_dist:
        return False
    if _bpm_fold(r.bpm, host.bpm) > cfg.max_bpm_fold:
        return False
    return r.recording_id != host.recording_id


def sample_window_v2(
    catalog: StemCatalog,
    *,
    mix_id: str,
    curriculum: str,
    rng: np.random.Generator,
    slot_base: int = 100,
) -> MashupWindowV2 | None:
    cfg = get_curriculum(curriculum)
    beds = _pick_beds(catalog, cfg, rng)
    if beds is None:
        return None

    window_s = cfg.window_s
    instrumentals = _schedule_instrumentals(beds, window_s, cfg, rng, slot_base)

    n_loops = cfg.n_loops
    if isinstance(n_loops, tuple):
        n_loops = int(rng.integers(n_loops[0], n_loops[1] + 1))
    n_acap = int(rng.integers(cfg.acap_count[0], cfg.acap_count[1] + 1))
    n_linear = max(0, n_acap - n_loops)
    if n_linear < 1 and n_loops < 1:
        return None

    acappellas: list[AcappellaSpan] = []
    used_payloads: set[str] = set()
    w_idx = 1

    # Loop spans first (fixed structure).
    for _ in range(n_loops):
        t = float(rng.uniform(window_s * 0.15, window_s * 0.55))
        host = _host_at(t, instrumentals)
        pool = _pick_payloads(catalog, host.bed, cfg, 8, rng)
        if pool is None:
            continue
        payload = next((p for p in pool if p.recording_id not in used_payloads), pool[0])
        used_payloads.add(payload.recording_id)
        slot = f"{host.slot_label}w{w_idx}"
        w_idx += 1
        ac = _loop_acap(payload, host.bed, t, window_s, cfg, slot, host.slot_label, rng)
        if ac:
            acappellas.append(ac)

    # Linear overlays with overlap bias.
    for _ in range(n_linear):
        t = float(rng.uniform(8.0, window_s * 0.75))
        host = _host_at(t, instrumentals)
        pool = _pick_payloads(catalog, host.bed, cfg, 6, rng)
        if pool is None:
            continue
        payload = next((p for p in pool if p.recording_id not in used_payloads), None)
        if payload is None:
            continue
        used_payloads.add(payload.recording_id)
        dur = float(rng.uniform(*cfg.acap_duration_s))
        slot = f"{host.slot_label}w{w_idx}"
        w_idx += 1
        ac = _linear_acap(payload, host.bed, t, dur, window_s, slot, host.slot_label, rng)
        if ac:
            acappellas.append(ac)

    if len(acappellas) < max(3, cfg.acap_count[0] // 2):
        return None

    # Regular (full-song) plays — data-limited; skipped silently when the
    # dual-stem catalog is too small (only tracks with BOTH stems qualify).
    n_regulars = cfg.n_regulars
    if isinstance(n_regulars, tuple):
        n_regulars = int(rng.integers(n_regulars[0], n_regulars[1] + 1))
    regulars: list[RegularSpan] = []
    used_regulars: set[str] = set()
    for _ in range(n_regulars):
        if not catalog.regulars:
            break
        t = float(rng.uniform(8.0, window_s * 0.7))
        host = _host_at(t, instrumentals)
        cands = [
            r
            for r in catalog.regulars
            if r.recording_id not in used_regulars
            and _regular_compatible(r, host.bed, cfg)
        ]
        if not cands:
            continue
        reg = cands[int(rng.integers(0, len(cands)))]
        used_regulars.add(reg.recording_id)
        dur = float(rng.uniform(*cfg.acap_duration_s))
        slot = f"{host.slot_label}w{w_idx}"
        w_idx += 1
        rs = _regular_span(reg, host.bed, t, dur, window_s, slot, host.slot_label, rng)
        if rs:
            regulars.append(rs)

    return MashupWindowV2(
        mix_id=mix_id,
        window_duration_s=window_s,
        instrumentals=instrumentals,
        acappellas=tuple(acappellas),
        curriculum=curriculum,
        regulars=tuple(regulars),
    )
