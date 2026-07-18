"""TRM recursive core (bake-off §2.1) + sim->offset-evidence gather (§2.2/2.3).

Tensor-only: no audio, no GT. Proves the recursion wiring (shape, gradient
through ALL T inner steps, can overfit a tiny fixed batch) before it ever
touches a real span — the v0 sanity, in unit form."""

from __future__ import annotations

import torch

from workspaces.alignment_prototype.trajectory import trm
from workspaces.alignment_prototype.trajectory.offset_coords import OffsetVocab


def _seed():
    torch.manual_seed(0)


def test_forward_shapes_deep_supervision_steps():
    _seed()
    B, N, d, V = 2, 12, 16, 9
    core = trm.TRMCore(d_model=d, vocab_size=V, n_sup=3, n_inner=2, T=2)
    x = torch.randn(B, N, d)
    out = core(x)
    assert len(out.logits) == 3  # one logit grid per supervision step
    assert out.logits[-1].shape == (B, N, V)
    assert out.q[-1].shape == (B,)  # per-example ACT halt signal


def test_gradient_reaches_net_params_through_the_grad_round():
    _seed()
    B, N, d, V = 2, 8, 16, 7
    core = trm.TRMCore(d_model=d, vocab_size=V, n_sup=1, n_inner=3, T=2)
    x = torch.randn(B, N, d)
    out = core(x)
    target = torch.randint(0, V, (B, N))
    loss = torch.nn.functional.cross_entropy(
        out.logits[-1].reshape(-1, V), target.reshape(-1)
    )
    loss.backward()
    grads = [p.grad for p in core.net.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_can_overfit_a_tiny_fixed_batch():
    _seed()
    B, N, d, V = 2, 6, 24, 5
    core = trm.TRMCore(d_model=d, vocab_size=V, n_sup=2, n_inner=3, T=2)
    x = torch.randn(B, N, d)
    target = torch.randint(0, V, (B, N))
    opt = torch.optim.Adam(core.parameters(), lr=5e-3)

    def step():
        opt.zero_grad()
        out = core(x)
        loss = sum(
            torch.nn.functional.cross_entropy(lg.reshape(-1, V), target.reshape(-1))
            for lg in out.logits
        ) / len(out.logits)
        loss.backward()
        opt.step()
        return loss.detach().item()

    first = step()
    for _ in range(120):
        last = step()
    assert last < first * 0.5  # recursion actually learns the fixed answer


def test_y_mode_every_differs_from_once():
    _seed()
    x = torch.randn(1, 6, 16)
    once = trm.TRMCore(d_model=16, vocab_size=5, y_mode="once", n_sup=1, n_inner=2, T=2)
    every = trm.TRMCore(
        d_model=16, vocab_size=5, y_mode="every", n_sup=1, n_inner=2, T=2
    )
    every.load_state_dict(once.state_dict())  # same weights, different recursion
    assert not torch.allclose(once(x).logits[-1], every(x).logits[-1])


def test_trm_decoder_is_a_dropin_over_sim_emitting_offset_logits():
    _seed()
    B, Tm, Tr = 2, 10, 40
    vocab = OffsetVocab(lo_bin=-4, hi_bin=60, res_bin=1)
    dec = trm.TRMDecoder(vocab, d_model=16, n_sup=2, n_inner=2, T=2)
    sim = torch.randn(B, 2, Tm, Tr)
    feat_kind = torch.zeros(B, dtype=torch.long)
    ref_valid = torch.ones(B, Tr, dtype=torch.bool)
    out = dec(sim, feat_kind, ref_valid)
    assert out.logits[-1].shape == (B, Tm, vocab.size)
    # an offset-target CE is well-defined against the emitted logits
    target = torch.randint(0, vocab.size, (B, Tm))
    loss = torch.nn.functional.cross_entropy(
        out.logits[-1].reshape(-1, vocab.size), target.reshape(-1)
    )
    assert torch.isfinite(loss)


def test_logits_stay_bounded_at_init_despite_deep_recursion():
    # regression: v0 saw CE ~8e7 from an exploding residual stream over
    # n_sup*T*n_inner net applications; the answer-latent LayerNorm bounds it.
    _seed()
    core = trm.TRMCore(d_model=64, vocab_size=1666, n_sup=3, n_inner=6, T=3)
    x = torch.randn(2, 40, 64)
    out = core(x)
    assert torch.isfinite(out.logits[-1]).all()
    assert out.logits[-1].abs().max().item() < 50.0


def test_offset_targets_from_ref_bin_targets():
    # frame t plays ref bin (t + 10) -> offset 10; one null; one ignore(-1)
    Tm = 5
    target_idx = torch.tensor([[10, 11, -1, 13, 14]])  # ref bins; -1 = ignore
    target_null = torch.tensor([[False, False, False, True, False]])
    mix_valid = torch.ones(1, Tm, dtype=torch.bool)
    vocab = OffsetVocab(lo_bin=-4, hi_bin=60, res_bin=1)
    labels = trm.trm_offset_targets(target_idx, target_null, mix_valid, vocab)
    # frames 0,1 -> offset 10 -> class (10 - lo_bin)
    assert labels[0, 0].item() == 10 - vocab.lo_bin
    assert labels[0, 1].item() == 11 - 1 - vocab.lo_bin  # 11-frame1=10
    from workspaces.alignment_prototype.trajectory.offset_coords import IGNORE_INDEX

    assert labels[0, 2].item() == IGNORE_INDEX
    assert labels[0, 3].item() == vocab.null_index  # null wins over position


def test_trm_decode_segments_from_a_peaked_offset_logit():
    Tm = 12
    vocab = OffsetVocab(lo_bin=-4, hi_bin=40, res_bin=1)
    logits = torch.full((1, Tm, vocab.size), -10.0)
    off_idx = 20 - vocab.lo_bin  # constant offset 20
    logits[0, :, off_idx] = 10.0
    out = trm.TRMOutput(logits=[logits], q=[torch.zeros(1)], y=logits, z=logits)
    segs = trm.trm_decode_segments(out, bin_s=0.5, vocab=vocab)
    assert len(segs) == 1
    mix0, ref0, _ = segs[0]
    assert mix0 == 0.0
    assert abs(ref0 - 20 * 0.5) < 0.5  # offset 20 bins == 10 s ref start


def test_sim_to_offset_evidence_gathers_the_diagonal():
    # a sim grid with a bright diagonal at ref = t + 4 should light up the
    # offset column for off_bin == 4 in the fixed offset-evidence grid.
    Tm, Tr = 6, 12
    sim = torch.zeros(1, 2, Tm, Tr)
    for t in range(Tm):
        sim[0, 0, t, t + 4] = 1.0
    vocab = OffsetVocab(lo_bin=0, hi_bin=8, res_bin=1)
    ev = trm.sim_to_offset_evidence(sim, vocab)  # (B, 2, Tm, n_offsets)
    assert ev.shape == (1, 2, Tm, vocab.n_offsets)
    col = ev[0, 0, :, 4]  # offset index 4 == off_bin 4 (lo_bin=0)
    assert torch.allclose(col, torch.ones(Tm))
