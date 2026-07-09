"""Dead argparse-flag guardrail: catch declared-but-unread flags (`--fiber-k`,
`--tol-s`) without false-positiving on the patterns that legitimately read a
flag under a different name."""

from __future__ import annotations

from pathlib import Path

from scripts.guardrails import _check_dead_argparse_flags

_P = Path("dummy.py")


def _codes(src: str) -> list[str]:
    return [v.detail for v in _check_dead_argparse_flags(_P, src)]


def test_declared_but_unread_is_flagged():
    src = (
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--fiber-k', type=int, default=6)\n"
        "args = p.parse_args()\n"
        "print('hello')\n"
    )
    out = _check_dead_argparse_flags(_P, src)
    assert len(out) == 1
    assert out[0].check == "dead_flag"
    assert "--fiber-k" in out[0].detail


def test_read_flag_is_clean():
    src = (
        "p.add_argument('--fiber-k', type=int, default=6)\n"
        "args = p.parse_args()\n"
        "print(args.fiber_k)\n"
    )
    assert _codes(src) == []


def test_dest_redirect_is_clean():
    # --no-fibers stores into `fibers`; args.fibers is read.
    src = (
        "p.add_argument('--fibers', action='store_true', default=True)\n"
        "p.add_argument('--no-fibers', dest='fibers', action='store_false')\n"
        "args = p.parse_args()\n"
        "if args.fibers:\n    pass\n"
    )
    assert _codes(src) == []


def test_access_under_other_variable_name_is_clean():
    # A helper returns the namespace; the caller reads it as `a`, not `args`.
    src = (
        "def parse(argv):\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--audio-root', type=str)\n"
        "    return p.parse_args(argv)\n"
        "a = parse(None)\n"
        "run(a.audio_root)\n"
    )
    assert _codes(src) == []


def test_dynamic_access_is_skipped():
    # vars(args)/**kwargs can read any flag — don't guess.
    src = (
        "p.add_argument('--never-read-directly', type=str)\n"
        "args = p.parse_args()\n"
        "cfg = Config(**vars(args))\n"
    )
    assert _codes(src) == []


def test_positional_arg_not_flagged():
    src = (
        "p.add_argument('paths', nargs='+')\n"
        "args = p.parse_args()\n"
        "print('no attr access at all')\n"
    )
    assert _codes(src) == []
