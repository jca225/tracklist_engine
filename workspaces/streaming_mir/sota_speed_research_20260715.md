# SOTA speed research — deep-research harness verdicts (2026-07-15)

105 agents, 3-vote adversarial verification per claim. Question: fastest possible pipeline at the ≥38 dB near-identical bar.

## Summary

No verified 2024–2026 result offers a drop-in, equal-quality (≥38 dB near-identical) replacement for the BS-RoFormer + 2x Mel-Band RoFormer ensemble: the efficient successors that survived verification are parameter-efficiency plays (Moises-Light: 13x fewer params at ~equal MUSDB SDR but no code, no checkpoints, no inference timings; SCNet: 2x CPU speedup vs HTDemucs but below BS-RoFormer quality; Windowed Sink Attention: 44.5x attention-compute reduction but a real −0.95 dB SDR loss and, per its own authors, only modest wall-clock gains on high-end GPUs at 8 s chunks). The highest-expected-impact, lowest-risk lever on the 86% separation share is therefore in-house: (a) ensemble reduction — the MSST repo itself publishes zero quantified SDR gain for multi-checkpoint ensembles over the best single model, and newer single checkpoints (BS PolarFormer 11.00, MelBand-Kim 10.98) already edge out the classic viperx BS-RoFormer (10.87) on MSST's own leaderboard, so a 5→2-pass A/B against the 38 dB bar is well-motivated; and (b) fp16/SDPA inference engineering — RoFormer-style band-transformers dispatch through F.scaled_dot_product_attention by default in the reference implementation, but the true flash kernel is force-disabled on A10G/4090 (only A100 sm80 enables it), leaving verified kernel headroom. For embeddings, distilled MERT students measurably lose music-task accuracy (SingerID 85.69→78.36) so 4x compression is not quality-free, and MuQ is quality-superior but same-size (310M) and not faster — no validated fast path to MERT-330M layer-6 equivalence exists. No claim about replacing Essentia's key/BPM/classifier stack on GPU survived verification.

## Verified findings

### [0] (high, vote 3-0 (x2, merged claims 0+1))

Best single-model efficiency result (Q1): Moises-Light, a band-split U-Net (arXiv 2510.06785, Oct 2025), slightly beats BS-RoFormer's no-extra-data MUSDB18-HQ average (9.96 vs 9.80 dB cSDR) with ~13x fewer parameters per stem (5M vs 72M) — but the paper publishes NO inference speed/FLOPs/RTF numbers and NO code or checkpoints (only an unofficial, weights-free reimplementation exists), so it is architecture-efficiency evidence, not a deployable speedup. It is also compared against the MUSDB-only 9.80 BS-RoFormer config, far below the extra-data ep368-class checkpoints (~12 dB filename SDR) the pipeline actually runs, and its bass stem is 1.27 dB worse.

**Evidence:** Table 4: Moises-Light avg 9.96 dB (vocals 10.92, drums 10.93, bass 10.04, other 7.95) vs BS-RoFormer avg 9.80 (10.66/9.49/11.31/7.73); params 5M x4 vs 72M x4. Verifier confirmed no inference-cost measurements and no official release anywhere in the paper; only unofficial github.com/crlandsc/moises-light exists, without trained weights.

**Sources:** https://arxiv.org/html/2510.06785v1

### [1] (high, vote 3-0 (x2, merged claims 2+3))

Attention-efficiency work does not help this pipeline (Q1/Q2): Windowed Sink Attention in Mel-Band RoFormer (Smule Labs, arXiv 2510.25745, code+checkpoints MIT) cuts attention computations 44.5x but retains only 92% of SDR (11.17 vs 12.12 dB median MUSDB18HQ, −0.95 dB; cSDR 9.89→8.60, Bleedless 41.0→30.1) — an objectively non-identical output that fails the ≥38 dB near-identical bar. The authors themselves state wall-clock speedups are 'modest' on modern high-end GPUs at the standard 8-second window, with wins only on >4,000-frame sequences or low-parallelism consumer devices — i.e., minimal end-to-end impact on a chunk-batched A10G/4090 pipeline even if the quality loss were acceptable.

**Evidence:** Verbatim: 'The WSA model achieves a median of 11.17 dB SDR compared to 12.12 dB for the original MB-R, retaining 92% (or 0.95 dB less) of the performance' and 'For the 8-second inference window, modern high-end GPUs can effectively parallelize the original quadratic attention, making wall-clock speedups modest.' The 44.5x is attention ops only (641,601 → 14,418), not total model FLOPs.

**Sources:** https://arxiv.org/pdf/2510.25745, https://github.com/smulelabs/windowed-roformer

### [2] (high, vote 3-0 (x3, merged claims 4+5+8))

SCNet is an efficiency play below the quality bar (Q1): SCNet (arXiv 2401.13276) reaches 9.0 dB avg SDR on MUSDB18-HQ without extra data at 48% of HTDemucs's CPU inference time (~2x), but 9.0 dB (and even SCNet-Large's 9.69/starrytong 9.70) is below BS-RoFormer's 9.80 no-extra-data figure; its speed comparison is CPU-vs-HTDemucs, not GPU-vs-RoFormer. On the MSST registry SCNet XL scores MUSDB test avg 9.80 / vocals 11.05 (XL IHF: 10.08 / 11.42) — competitive on vocals but measured on MUSDB test while the top RoFormer vocal checkpoints are scored on the Multisong dataset, so cross-benchmark comparison cannot establish equal quality. Not a drop-in replacement for the ensemble.

**Evidence:** SCNet abstract: '9.0 dB SDR on MUSDB18-HQ... without using extra data' and 'SCNet's CPU inference time is only 48% of HT Demucs'; BS-RoFormer abstract: 9.80 dB no-extra-data. MSST doc: 'SCNet XL | MUSDB test avg: 9.80 (bass: 9.23, drums: 11.51 vocals: 11.05 other: 7.41)'; RoFormer vocal rows are explicitly 'measured on Multisong Dataset'.

**Sources:** https://arxiv.org/abs/2401.13276, https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md, https://arxiv.org/abs/2309.02612

### [3] (high, vote 3-0 (x2, merged claims 6+7))

Checkpoint landscape supports ensemble slimming (Q1/Q3): on MSST's independent Multisong leaderboard, the top single vocal checkpoints sit within 0.13 dB of each other — BS PolarFormer 11.00 > MelBand RoFormer (KimberleyJensen) 10.98 > BS-RoFormer viperx ep_317 10.87 — so newer single checkpoints already exceed the classic viperx model that anchors many MSST-style ensembles. Caveat: filename SDRs (e.g. '12.9755' in the viperx checkpoint name, sibling of ep_368) are the author's own eval on a different set and are NOT comparable to the leaderboard's 10.87.

**Evidence:** Leaderboard rows verified verbatim: 'BS PolarFormer | vocals / other | SDR vocals: 11.00', 'MelBand Roformer (KimberleyJensen edition) | 10.98', 'BS Roformer (viperx edition) | 10.87' (weights file model_bs_roformer_ep_317_sdr_12.9755.ckpt). Note margins are ~0.1 dB and the pipeline's ep368 checkpoint is the ep_317 sibling, not the exact listed file.

**Sources:** https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md

### [4] (high, vote 3-0 (claim 9); related quantified-ensemble claims refuted 0-3)

Ensemble reduction has no published counter-evidence (Q3): the MSST repo — the stack the pipeline is modeled on — publishes NO quantified SDR gain for multi-checkpoint ensembles over the best single model. Its only empirical statements are that avg_wave is the best merge method and that ensembling helps only between models of roughly equal quality. There is therefore no measured published justification for 5 passes vs 2; combined with the ≤0.13 dB spread among top single checkpoints, a 5→2 (or 5→1) reduction is the largest-expected-impact lever on the 86% separation share (~2.5x on that stage) — but it MUST be validated in-house against the 38 dB near-identical bar, because attempts to cite quantified ensemble-gain numbers from the literature (SDX'23 paper, arXiv 2305.07489) were all refuted in verification.

**Evidence:** docs/ensemble.md verbatim: 'In my experiments avg_wave was always better or equal in SDR score comparing with other methods' and 'It's better to ensemble models which are of equal quality'; zero ensemble-vs-best-single SDR numbers anywhere in the repo (README, pretrained_models.md, mel_roformer_experiments.md all checked).

**Sources:** https://github.com/ZFTurbo/Music-Source-Separation-Training, https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/ensemble.md

### [5] (high, vote 3-0 (x2, merged claims 12+13))

SDPA is already the structural default in the reference RoFormer implementations, but the flash kernel is disabled on A10G/4090 (Q2): lucidrains' BS-RoFormer and Mel-Band RoFormer default flash_attn=True and dispatch through F.scaled_dot_product_attention with plain dense non-causal attention (RoPE applied to q/k before the call) — so these architectures are SDPA-compatible with no quality-changing modification. However, the Attend module force-enables the true flash backend ONLY on compute capability 8.0 (A100); on A10G (sm86) and 4090 (sm89) it sets FlashAttentionConfig(False, True, True), i.e. math/mem-efficient backends only. Removing the legacy sdp_kernel gating (or upgrading the attention path) on the deployed stack is verified, concrete headroom — the deployed MSST/ZFTurbo code vendors derived attention code, so the exact shipped path must be checked.

**Evidence:** bs_roformer.py line ~302: 'flash_attn = True' default; attend.py: 'out = F.scaled_dot_product_attention(q, k, v, ...)' and 'if device_properties.major == 8 and device_properties.minor == 0: ... FlashAttentionConfig(True, False, False) ... else: ... FlashAttentionConfig(False, True, True)' wrapped in torch.backends.cuda.sdp_kernel. A10G=sm_86, 4090=sm_89 both hit the else branch.

**Sources:** https://github.com/lucidrains/BS-RoFormer

### [6] (high, vote 3-0 (x2, merged claims 10+11); MSST fp16-flag claims refuted 0-3)

Quantized RoFormer inference exists but ships zero quality evidence (Q2): BSRoformer.cpp (GGML C++ engine for BS-RoFormer and Mel-Band RoFormer, v0.1.0, June 2026) supports FP32/FP16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1 weights (K-Quants unsupported by the converter) and recommends Q8_0 as the precision/performance balance — but the project publishes NO measured speed, RTF, memory, or SDR numbers vs PyTorch fp32 (only element-wise activation-diff pass criteria: FP16 <5e-4, Q8_0 <5e-3 on a ~3 s segment). It provides zero evidence for or against the ≥38 dB bar; adoption would require full in-house SDR validation. Notably, NO verified practitioner-shipped fp16/torch.compile/TensorRT speed+quality measurements for these exact architectures survived — the two claims about MSST's own AMP/flags were both refuted.

**Evidence:** README verbatim: 'Multiple Quantization Support: FP32/FP16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1'; q8_0 marked 'Recommended (balance of precision and performance)'; 'The conversion script currently does not support K-Quant types'. Verifier confirmed no published benchmarks anywhere in the repo (27 stars, young project — low ecosystem maturity).

**Sources:** https://github.com/chenmozhijin/BSRoformer.cpp

### [7] (high, vote 3-0 (x2, claims 14+15) + 2-1 (claim 18, merged))

Distilled MERT is not quality-free (Q4): DistilHuBERT-style compression of MERT-base (95M, mert-v0-public) to a 23M 2-layer student causes measurable music-task losses — SingerID 85.69→78.36, PitchID 91.26→87.21, ESC50 74.00→67.90 (arXiv 2505.13270); the joint HuBERT+MERT multi-distilled 23M student (arXiv 2506.07237) does better, retaining near-teacher averages (non-ASR avg 80.40 vs 81.16 for a 46M ensemble; vs the MERT teacher: SingID 84.27 vs 85.69, PID 87.55 vs 91.26). Critically, both papers distill against teacher layers 4/8/12 with L1+cosine loss and never target or evaluate MERT layer-6 feature equivalence — the layer this pipeline uses — and both use mert-v0-public, not MERT-v1-95M/330M. So ~4x compression is possible with small downstream-task cost, but layer-6 feature-level equivalence is entirely unestablished.

**Evidence:** 2506.07237 Table II + Eq.(2) (loss summed over l∈{4,8,12}, DistilHuBERT L1+cosine); verbatim: 'merely a slight drop of 0.8% (80.40% vs. 81.16%)... reduce the number of parameters by 50% (from 46M to 23M)'. 2505.13270 Table 1: distil-MERT 23M SingerID 78.36 vs MERT 95M 85.69. Caveats: 2505.13270's student was distilled on LibriSpeech speech data (adverse config); the 2-layer students structurally have no layer 6; both averages mix speech and music tasks.

**Sources:** https://arxiv.org/pdf/2506.07237, https://arxiv.org/pdf/2505.13270

### [8] (high, vote 3-0 (x2, merged claims 16+17))

MuQ is a quality upgrade to MERT-330M, not a speed lever (Q4): MuQ (arXiv 2501.01108, Tencent) outperforms MERT-v1-330M on the MARBLE average (76.7, iterative 77.0, vs 74.4 across 9 tasks) but is ~310M parameters — the same size class — and the paper reports no inference-speed, latency, or FLOPs advantage. Adopting it would not reduce the ~6% MERT wall-clock share (it would likely increase it vs the current 95M model), and average MARBLE superiority does not establish drop-in equivalence to MERT layer-6 features.

**Evidence:** Main MARBLE table: MERT-330M avg 74.4, MuQ 76.7, MuQ_iter 77.0; paper: 'We stack 12 layers of Conformer in MuQ, with 310M parameters in total'; full-text scan found zero inference-efficiency metrics. Self-reported numbers on a standard public benchmark, corroborated by independent 2025-26 papers treating MuQ as competitive-to-stronger.

**Sources:** https://arxiv.org/abs/2501.01108

## Refuted claims (killed by verifiers — do not act on these)

- **(0-3)** MSST's inference CLI (utils/settings.py) exposes no half-precision, torch.compile, TensorRT, or quantization flags; the only shipped quality/speed trade-off knobs at inference are TTA and time-shift averaging, and the documented --use_tta option costs 3x runtime for only a slight quality gain — i.e. the stack the user runs ships essentially unoptimized fp32 inference, leaving headroom for fp16/SDPA work.
  - source: https://github.com/ZFTurbo/Music-Source-Separation-Training

- **(0-3)** MSST's own BS-RoFormer config documentation ships mixed-precision float16 enabled by default (use_amp: true) but flash attention disabled (flash_attn: false) — so the maintainer treats fp16 AMP as safe for these RoFormer architectures while FlashAttention is present as a config flag but not the shipped default.
  - source: https://github.com/ZFTurbo/Music-Source-Separation-Training

- **(0-3)** On MSST's pretrained-model leaderboard the top single-checkpoint vocal models are within ~0.13 dB SDR of each other (BS PolarFormer 11.00, Mel-Band RoFormer Kim 10.98, BS-RoFormer viperx 10.87), implying the marginal SDR available from ensembling extra RoFormer checkpoints on top of the best one is small relative to a 38 dB near-identity bar's tolerance.
  - source: https://github.com/ZFTurbo/Music-Source-Separation-Training

- **(0-3)** Blending multiple models was the dominant strategy among top SDX'23 music-demixing solutions, but the field has no principled guidance on ensemble construction — the paper explicitly states no clear guidelines exist for building an effective ensemble, and it provides no controlled comparison of an ensemble against the same team's best single model.
  - source: https://transactions.ismir.net/articles/10.5334/tismir.171

- **(0-3)** Multi-model separation ensembles outperform the best single model by only a small SDR margin at considerable computational cost: on the Synth MVSep benchmark the best single model scores 11.11 dB instrumental / 11.40 dB vocal SDR, while the best 4-model ensemble (3 MDX + 1 Demucs4) scores 11.26 / 11.61 — a gain of roughly +0.15/+0.21 dB for 4x the passes.
  - source: https://arxiv.org/pdf/2305.07489

- **(0-3)** A 2-model ensemble can score WORSE than the best single model: in Table 2, the ensemble 'UVR-MDX-NET Inst 3 and htdemucs_ft' achieves 10.88 dB instrumental / 11.20 dB vocal SDR, below the best single model's 11.11 / 11.40 (Table 1) — so reducing an ensemble to 2 checkpoints only preserves quality if the retained models are the strongest ones, not merely fewer passes.
  - source: https://arxiv.org/pdf/2305.07489
