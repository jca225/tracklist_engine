"""BB12-realistic curriculum configs and section sampling priors."""

from __future__ import annotations

from dataclasses import dataclass

CURRICULUM_V2 = {
    "bb12-lite": {
        "window_s": 180.0,
        "n_instrumentals": 2,
        "acap_count": (5, 7),
        "n_loops": 1,
        "n_regulars": (0, 1),
        "instr_jump_segments": (2, 3),
        "instr_jump_prob": 0.85,
        "max_key_dist": 1,
        "max_bpm_fold": 0.05,
        "handoff_crossfade_s": 3.0,
        "acap_duration_s": (22.0, 42.0),
        "loop_phrase_s": (3.5, 5.5),
        "loop_repeats": (3, 5),
    },
    "bb12-med": {
        "window_s": 300.0,
        "n_instrumentals": 2,
        "acap_count": (8, 12),
        "n_loops": (1, 2),
        "n_regulars": (1, 2),
        "instr_jump_segments": (2, 4),
        "instr_jump_prob": 0.75,
        "max_key_dist": 2,
        "max_bpm_fold": 0.06,
        "handoff_crossfade_s": 4.0,
        "acap_duration_s": (25.0, 50.0),
        "loop_phrase_s": (3.5, 6.0),
        "loop_repeats": (4, 6),
    },
    "bb12-full": {
        "window_s": 480.0,
        "n_instrumentals": 3,
        "acap_count": (14, 20),
        "n_loops": (2, 3),
        "n_regulars": (2, 3),
        "instr_jump_segments": (2, 4),
        "instr_jump_prob": 0.85,
        "max_key_dist": 2,
        "max_bpm_fold": 0.08,
        "handoff_crossfade_s": 4.0,
        "acap_duration_s": (28.0, 55.0),
        "loop_phrase_s": (3.5, 6.0),
        "loop_repeats": (4, 7),
    },
}


@dataclass(frozen=True)
class CurriculumV2:
    name: str
    window_s: float
    n_instrumentals: int
    acap_count: tuple[int, int]
    n_loops: int | tuple[int, int]
    n_regulars: int | tuple[int, int]
    instr_jump_segments: tuple[int, int]
    instr_jump_prob: float
    max_key_dist: int
    max_bpm_fold: float
    handoff_crossfade_s: float
    acap_duration_s: tuple[float, float]
    loop_phrase_s: tuple[float, float]
    loop_repeats: tuple[int, int]


def get_curriculum(name: str) -> CurriculumV2:
    raw = CURRICULUM_V2.get(name, CURRICULUM_V2["bb12-lite"])
    n_loops = raw["n_loops"]
    return CurriculumV2(
        name=name,
        window_s=float(raw["window_s"]),
        n_instrumentals=int(raw["n_instrumentals"]),
        acap_count=tuple(raw["acap_count"]),
        n_loops=n_loops,
        n_regulars=raw.get("n_regulars", 0),
        instr_jump_segments=tuple(raw["instr_jump_segments"]),
        instr_jump_prob=float(raw["instr_jump_prob"]),
        max_key_dist=int(raw["max_key_dist"]),
        max_bpm_fold=float(raw["max_bpm_fold"]),
        handoff_crossfade_s=float(raw["handoff_crossfade_s"]),
        acap_duration_s=tuple(raw["acap_duration_s"]),
        loop_phrase_s=tuple(raw["loop_phrase_s"]),
        loop_repeats=tuple(raw["loop_repeats"]),
    )
