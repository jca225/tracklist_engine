"""GPU-free unit tests for the stem-MERT layer-slicing core (Phase 1B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alignment.stem_mert import (
    _measure_stack,
    slice_layers,
)


@dataclass
class _FakeMeasure:
    embedding_bytes: bytes
    dim: int


def _fake_measure(n_layers: int, dim: int, fill: float) -> _FakeMeasure:
    arr = np.full((n_layers, dim), fill, dtype=np.float16)
    # make each layer distinguishable: layer L gets value fill + L
    for ell in range(n_layers):
        arr[ell] = fill + ell
    return _FakeMeasure(embedding_bytes=arr.tobytes(), dim=dim)


def test_measure_stack_shape_and_layout():
    n_layers, dim = 25, 4
    ms = (_fake_measure(n_layers, dim, 0.0), _fake_measure(n_layers, dim, 100.0))
    stack = _measure_stack(ms, dim)
    assert stack.shape == (2, n_layers, dim)
    # measure 0, layer 3 -> value 3; measure 1, layer 22 -> 100+22
    assert np.allclose(stack[0, 3], 3.0)
    assert np.allclose(stack[1, 22], 122.0)


def test_slice_layers_picks_and_transposes():
    n_layers, dim = 25, 4
    ms = (_fake_measure(n_layers, dim, 0.0), _fake_measure(n_layers, dim, 0.0))
    stack = _measure_stack(ms, dim)
    got = slice_layers(stack, (3, 22))
    # (dim, n_measures) convention for chamfer
    assert got[3].shape == (dim, 2)
    assert got[22].shape == (dim, 2)
    assert np.allclose(got[3], 3.0)
    assert np.allclose(got[22], 22.0)


def test_measure_stack_empty():
    assert _measure_stack((), 4).shape == (0, 0, 4)
