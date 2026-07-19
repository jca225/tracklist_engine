# HANDOFF — GPU compute for synthetic-transfer scale-up (2026-07-17)

**For:** the agent running the GPU job. **From:** the synthetic-transfer spike session.
**One-line:** synthetic→real transfer is validated 🟢; now run it at scale on a GPU box.

---

## 1. Why (context)

North star: SOTA aligner across ~40k DJ sets. This session **pivoted** (decision
D13 in `docs/alignment_state_of_record.md`): PWS demoted to the fusion layer; the
primary bet is **synthetic-supervised learned placement/structure** — train a
placement model on *synthetic* (manufactured-label) mixes, validate on the 2 real
GT sets (BB11 `2nvzlh2k`, BB12 `1fsnxchk`).

**Validated result (🟢, leakage-clean):** trajectory decoder, held-out BB12:
- real-only training (ceiling) = **0.369** trajectory-acc
- real + synthetic augmentation = **0.393** (beats ceiling, more stable across epochs)
- leakage check: 0/66 synthetic track_ids in BB12 eval → transfer is genuine.

Committed: state-of-record SHAs `6906c48`, `74dfc70`, `75f9a94`; spec
`docs/superpowers/specs/2026-07-17-synthetic-transfer-spike-design.md`.

## 2. The two GPU jobs to run

**JOB 1 — Synthetic volume curve (DECISIVE — do first).**
The +0.024 lift is at only **100** synthetic mixes. The go/no-go for the whole
program: **does more synthetic → more accuracy?**
- Generate more synthetic mixes: `python -m workspaces.alignment_prototype.synthetic_mix.generate_v2 --n <N> --curriculum bb12-lite --out data/synthetic_mixes_v2_<N> --seed <s>` (needs the source stem catalog from pi-storage via `mashup_compat.pairs`/`PI_DB`).
- Retrain the trajectory decoder at N ∈ {100, 500, 1000, 2000, …}, eval held-out BB12, plot traj-acc vs N. If it keeps climbing → GREEN, scale the learned-aligner program.
- Also run: matched-epoch + multi-seed (kill single-epoch noise), the other direction (train BB12→eval BB11), and a **pure-synthetic-only** variant (train on synthetic *only*, no real) to measure the true transfer gap vs. augmentation.

**JOB 2 — Mix-side analysis of the 1,016 downloaded mixes.**
Co-training harvest needs analyzed mixes; only **4** are analyzed today
(`set_measures`=0 beat grids, `set_stems`=4). Run RoFormer stems + beats + MERT +
fingerprints on the 1,016 mixes (+ their refs). This is the repo's existing
`analysis` stack (`scripts/vast_loop.py` / `analysis.vast_worker` pattern, pulls
from pi-storage). Then the co-training seam (`workspaces/pws_aligner/cotrain_seam.py`)
can harvest pseudo-labels. **No new downloads needed** — the mixes are already on pi.

## 3. Infra — EC2 g5 in us-east-2 (NOT SageMaker)

Quota reality (checked 2026-07-17, personal acct `008971646190`):
| region | EC2 On-Demand G (vCPU) | SageMaker g5 train |
|---|---|---|
| us-east-1 (default) | **0** | 1 |
| **us-east-2** | **8** ✅ | 1 |
| us-west-2 | 0 | 1 |
Spot G = 0 everywhere → **on-demand only**.

- **Use EC2 `g5.xlarge` (A10G, 4 vCPU) or `g5.2xlarge` (8 vCPU) in us-east-2**, on a
  Deep Learning AMI (CUDA+PyTorch preinstalled). SSH in, `git clone`, run — same
  shape as the Vast workflow. **No container/estimator/S3/IAM-role needed.**
- SageMaker was rejected: g5 quota=1 but needs container+estimator+S3 staging+an
  IAM execution role (role creation was **classifier-blocked** — needs explicit
  user authorization if pursued).
- S3 bucket `jspace-torch-008971646190` exists if staging is wanted.
- `~1.00/hr on-demand (g5.xlarge). DESTROY when done (on-demand bills continuously).

## 4. Exact reproduction recipe (the run that gave 🟢)

```bash
# ceiling (A, real-only):
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
python -u -m workspaces.alignment_prototype.trajectory.train \
  --split set --train-set 2nvzlh2k --eval-set 1fsnxchk --device cuda
# augmented (B, real+synthetic):  add  --synthetic-root data/synthetic_mixes_v2
```
`train.py` prints a no-model control (floor) + eval every 10 epochs; scores with
`path_decode.trajectory_acc`. Synthetic is train-only; eval stays on real GT.

## 5. Gotchas learned this session (READ — they cost hours)

- **MPS hangs trajectory training on Mac.** On the GPU box use `--device cuda`
  (the hang was Mac-MPS-specific; CUDA should be fine).
- **Always `PYTHONUNBUFFERED=1` (or `python -u`).** stdout block-buffers to files
  → the log looks frozen for 40+ min while it's actually working. This caused
  repeated false "it's hung" diagnoses.
- **`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`** after first model fetch — a stale
  HF file-lock (from a killed process) hung startup at 0 CPU. Let HuBERT
  (`facebook/hubert-base-ls960`) download once on the fresh box, THEN set offline.
- **Feature cache** persists to `workspaces/alignment_prototype/.feat_cache/`
  (`*_hubertL9.npy`, `*_mel64.npy`, `*_chroma.npy`). Cold pass over 100 synthetic
  mixes = ~40 min on CPU; **GPU + persisted cache is the whole point.**
- The Mac was slow because of **CPU contention from parallel `race`/`infer`
  agents** — a dedicated GPU box eliminates it.

## 6. Data the box needs

- BB aligning folders (mix+refs+stems): `~/aligning/2nvzlh2k*`, `~/aligning/1fsnxchk*`
  (Mac-local) — or pull from pi-storage.
- Synthetic mixes: `data/synthetic_mixes_v2` (100 windows, Mac-local) — or
  regenerate on the box (needs pi-storage stem catalog).
- GT fixtures: `labeling/fixtures/id_maps/{2nvzlh2k,1fsnxchk}.json` + aligning manifests.
- HuBERT: `facebook/hubert-base-ls960` (HF, first-fetch).
- pi-storage over Tailscale for JOB 2 (the 1,016 mixes + refs).

## 7. Caveats on the 🟢 (rigor follow-ups = part of JOB 1)
Modest lift at 100 mixes; *augmentation* not *pure-synthetic*; single direction
(eval BB12); single seed. The volume curve + multi-seed + reverse direction +
pure-synthetic variant are the rigor items.
