# FP SOTA integration checkpoint

The strict two-set integration proof did **not** clear its pass criterion, so
this checkpoint makes no SOTA claim and does not update
`docs/alignment_status.md`.

What held:

- dense instrumental fingerprinting repaired the two targeted BB11
  decoder-wall cases;
- FP belief preference is gated to instrumental spans;
- live FP selection is likewise instrumental-only, preferring the
  instrumental landmark and falling back to the recording's regular landmark;
- lyrics and HuBERT now have explicit environment-controlled skips for focused
  local proofs.

What failed:

- making the dense landmark index available to regular/acappella spans caused
  widespread false FP overrides, so those lanes now abstain;
- rerunning the present agentic policy with lyrics/HuBERT deliberately skipped
  does not reproduce the existing baseline placements outside the
  instrumental lane. Therefore the fast run is not a valid whole-board SOTA
  replacement even though the targeted instrumental cases improve.

## Delta proof verdict

An explicit delta harness now preserves the canonical agentic baseline
byte-for-byte outside instrumental spans whose candidate winner includes FP.
That isolated proof also failed the two-set SOTA criterion: the known
decoder-wall repairs were outweighed by other instrumental FP regressions.

The changed-span audit found no prospective discriminator in the available
signals. Wins and regressions shared the same FP-only source and effectively the
same belief quality. Selecting only the wins would require a GT-tuned
direction/magnitude rule, so this experiment is closed rather than
cherry-picked.

Generated timelines and scorer output remain local under ignored `out/` paths.
