# Aligner Failure Tables — BB11 & BB12
Generated 2026-07-14 from `ws0-scorer-deinflation` worktree.
Timelines scored: `2nvzlh2k_predicted_timeline.json` / `1fsnxchk_predicted_timeline.json`.
GT: `bb11_ground_truth.yaml` (de1ce92) / `bb12_ground_truth.yaml` (de1ce92).

Axis legend:
- **IDENTITY** — wrong recording picked; GT track name shown, aligner grabbed something else.
- **SET_START** — placed at wrong mix position; error = pred_set_start − GT audible onset.
- **REF_OFFSET** — right spot in the mix, wrong position in the reference track; error = pred_ref_start − expected_ref_start.
- **TRAJECTORY** — placement correct (< 5 s), ref start close (< 5 s), but internal structure (loops / jumps / segments) not reproduced; traj-acc % given.
- **MISROUTE** — aligner routed to wrong stem (e.g. regular instead of acappella).

---

## BB12 — 1fsnxchk (Two Friends Big Bootie Mix Volume 12)

### Summary
- **Identity:** 127/152 (84%); 25 misses; 13 spans had no same-slot GT row.
- **Set placement:** median 4.3 s, p90 44.8 s; 53 % within 5 s, 79 % within 15 s (n=139).
- **Ref offset (straight clips, n=47):** median 14.0 s; acappella median 14.6 s, regular 61.4 s.
- **Trajectory (fiber-aware):** overall 40 % traj-acc; acappella 36 %, regular 53 %, instrumental 33 %; multiseg+loop 42 %.

**Three dominant failure patterns:**
1. **Acappella ref-offset at track start** — aligner places the acappella in the right mix window but grabs the intro/verse-1 of the reference instead of the chorus or late-verse entry the DJ used (5w2 −182 s, 13w3 −175 s, 7w3 −102 s, 41w2 −114 s). Pattern: pred_ref near 0 while GT is 100–200 s in.
2. **Set_start blown by hundreds of seconds on the Slide cluster** — slot 33w3 lands 746 s early (aligner treats it as a different appearance of Calvin Harris - Slide); slots 42w1/42w2/42w4 are correspondingly displaced late in the outro medley because the aligner ran out of budget. One wrong anchor poisons four surrounding spans.
3. **Identity failures in overlapping multi-layer windows** — 25 misses cluster at moments where 3+ tracks play simultaneously (slots 1/1w2, 4/4w2/4w3, 14w2/14w3, 41w2/42) and the MERT identity head picks a neighbor recording instead of the correct one.

### Worst-First Table (18 rows; 7 more below cutoff with |error| 25–38 s)

| Mix time (mm:ss–mm:ss) | Ableton track (GT name + stem) | Slot | Axis | What the aligner did vs GT |
|---|---|---|---|---|
| 58:43–60:17 | Calvin Harris Slide (Voclr studio acapella) · acappella | 33w3 | SET_START −746 s | Placed the acappella 746 s **too early** in the mix (pred 46:17, GT 58:43). The aligner latched onto an earlier Slide appearance or confusion with 42w4; this is the single largest placement error. In Ableton the clip should be at ~58:43 with ref starting at 0:18. |
| 13:42–13:58 | Victorious – Official Acapella-Vocal Track · acappella | 9w3 | SET_START −184 s | Placed 184 s **too early** (pred 10:38, GT 13:42); additionally ref_start is wrong by +95 s (pred 7.7 s vs GT 123 s). The aligner placed it in the NGHTMRE/Boombox block instead of its actual entry and grabbed the verse-1 ref position instead of the bridge. |
| 58:43–60:17 | Calvin Harris Slide (Voclr studio acapella) · acappella | 42w4 | SET_START +150 s | Placed 150 s **too late** (pred 61:13, GT 58:43) — mirror error of 33w3; the aligner emitted two conflicting appearances of Slide and this second one landed past the end of the mix. Ref start is also at 1.5 s vs GT 18.4 s. |
| 57:45–60:01 | Taylor Swift – Mean (Taylor's Version) (Acapella) · acappella | 42w1 | SET_START +117 s | Placed 117 s **too late** (pred 59:42, GT 57:45); this is a 3-segment multiseg (GT starts at ref 0:00). The aligner placed it well into the final medley block instead of the correct entry at 57:45, and the segment structure is not reproduced. |
| 7:25–8:00 | Take You There (A Cappella) – Sean Kingston · acappella | 5w2 | REF_OFFSET −182 s | Placed correctly in mix (off by only 0.4 s), but aligner grabbed **ref position 0:16** while GT is at **3:18** (182 s into the reference). The acappella's intro was used instead of the later chorus section; in Ableton slide the ref-start marker from 0:16 to 3:18. |
| 18:39–18:55 | Two Friends – Pacific Coast Highway (Acapella) · acappella | 13w3 | REF_OFFSET −175 s | Mix placement is correct (off 2.5 s), but pred_ref is 0.7 s vs GT 178 s. The aligner grabbed the very start of the reference when the DJ was using a late-song section at ~2:58. Slide the ref-start marker to ~2:58. |
| 42:53–43:28 | Calvin Harris – Sweet Nothing (Remix) · regular | 31w3 | REF_OFFSET +135 s | Placed correctly in mix (off 3.2 s), but pred_ref is 255.6 s vs GT 105.1 s — the aligner jumped to a section ~4:15 into the track while the GT plays from ~1:45. |
| 21:15–21:32 | Kesha – Die Young (Studio Acapella) · acappella | 15w3 | REF_OFFSET +115 s | Placed 9.3 s too early in the mix; ref is off by +115 s (pred 111.9 s vs GT 6.4 s). The aligner grabbed a mid-track position when the DJ used the very opening of the acappella. Slide ref-start from ~1:52 back to ~0:06. |
| 56:51–57:13 | Good Time – Owl City ft. Carly Rae Jepsen (Acapella) · acappella | 41w2 | SET_START +53 s / REF_OFFSET −114 s | Both axes broken: placed 53 s late AND pred_ref 0.2 s vs GT 61.2 s. The aligner both missed the entry window and grabbed the intro of the reference instead of the mid-song section. Identity also wrong (pred is "Owl City & Carly Rae Jepsen – Good Time" which is the original, not the acappella). |
| 10:52–11:15 | Drake – Forever Acapella – ft. Kanye, Lil Wayne & Eminem · acappella | 7w3 | REF_OFFSET −102 s | Mix placement close (off 2.9 s), but pred_ref 100.8 s vs GT 199.4 s (Eminem verse at ~3:19 in the reference). The aligner landed on the Drake verse instead of the Eminem outro. |
| 33:11–34:15 | DVBBS & Jay Hardway – Voodoo · regular | 24 | REF_OFFSET +98 s | Mix placement correct (off 2.9 s). Pred_ref 126.3 s vs GT 31.7 s — aligner jumped to a late-track drop instead of the intro section the DJ actually played. |
| 52:28–53:00 | Swedish House Mafia – Save The World · instrumental | 38w3 | REF_OFFSET −96 s | Mix placement correct (off 2.1 s). Pred_ref 0.1 s vs GT 98.6 s; aligner started the instrumental at the very beginning instead of the ~1:39 mark. |
| 44:49–45:19 | Jordi Rivera & Sonny Bass – Bubblegum · regular | 32w3 | REF_OFFSET +94 s | Placed correctly (off 3.9 s), but pred_ref 168.5 s vs GT 78.7 s — the aligner played a later section (~2:49) when the DJ used the 1:19 mark. |
| 1:04–1:34 | Post Malone – Congratulations (Acapella) · acappella | 1w1 | REF_OFFSET −78 s | Mix placement correct (off 0.4 s). Pred_ref 92.4 s vs GT 170 s; the aligner grabbed the post-chorus section (~1:32) when the DJ used the late chorus at ~2:50. |
| 57:58–59:32 | The Chainsmokers – Honest (SAVI Remix) · regular | 42w5 | REF_OFFSET +78 s | Placed correctly (off 1.9 s), but pred_ref 120.7 s vs GT 0.1 s — aligner started ~2:01 into the track when the DJ started at the very top. |
| 47:39–47:57 | The Chainsmokers – KANYE ft. SirenXX (Acapella) · acappella | 34w3 | SET_START −21 s / REF_OFFSET +77 s | Placed 21 s early in the mix (pred 47:18, GT 47:39); ref is also off by +77 s (pred 91 s vs GT 35 s). Double error: early mix entry and wrong reference section. |
| 56:13–57:40 | Lucas & Steve & Mike Williams x Curbi – Let's Go · regular | 41 | SET_START +38 s / REF_OFFSET −75 s | Placed 38 s late (pred 57:37, GT 56:13) and ref off by −75 s (pred 1 s vs GT 38 s). The aligner stacked the entry into the outro medley instead of its correct slot, and grabbed the intro of the ref instead of the 0:38 section. |
| 58:04–60:58 | Macy Gray – I Try (Acapella) · acappella | 42w2 | SET_START +69 s | Placed 69 s too late (pred 59:13, GT 58:04); this is a 2-segment multiseg (GT ref starts at 35.9 s). The aligner missed the entry window and the segment structure is not reproduced. |

*Below cutoff (|error| 25–65 s): 20w3 (Disclosure – Latch acap, −67 s SET_START), 25w2 (Robin S – Show Me Love acap, −65 s SET_START), slot 2 (Manse instrumental, +65 s SET_START), 4w3 (Chromeo – Jealous acap, −65 s SET_START), 25w1 (Notorious B.I.G. acap, −61 s SET_START), 27 (RetroVision – Sunday, +61 s REF_OFFSET), 2w3 (Outside acap, +64 s REF_OFFSET).*

---

## BB11 — 2nvzlh2k (Two Friends Big Bootie Mix Episode 11)

### Summary
- **Identity:** 127/150 (85%); 23 misses; 8 spans had no same-slot GT row.
- **Set placement:** median 5.0 s, p90 45.8 s; 50 % within 5 s, 67 % within 15 s (n=143).
- **Ref offset (straight clips, n=53):** median 7.7 s; acappella median 12.1 s, regular 13.1 s, instrumental 0.9 s (instrumental is the cleanest axis).
- **Trajectory (fiber-aware):** overall ~35 % traj-acc; acappella 29 %, regular 44 %, instrumental 51 %; multiseg+loop 39 %. Strict (no fibers): acappella 12 %, regular 11 %, instrumental 27 %.

**Three dominant failure patterns:**
1. **Ref-offset at wrong song section** — aligner places the track correctly in the mix but grabs the intro/start of the reference when the DJ entered mid-song (36w1 −143 s, 11w4 −124 s, 28w2 +109 s, 39w5 −92 s, 23w1 −78 s, 12w3 +75 s). Affects both regular and acappella.
2. **Large set_start errors on medley/multi-layer slots** — 23w2 (+106 s), slot 35 (+100 s), slot 11 (+90 s), 37w2 (+89 s), 15w3 (+86 s), 8w2 (+85 s) are all off by 85–106 s. These correspond to complex beds or sections with heavy layering where the MERT sequence-placement jumps to the wrong region.
3. **Identity failures at rapid-transition acappella slots** — 23 misses; typical pattern is the aligner returns a neighbor acappella that shares MERT space (e.g. slot 6w3 predicts Kanye – Good Life when GT is Thomas Hayes – Neon / 3 Doors Down acap; slot 8w2 predicts Whethan – Savage when GT is Galantis – No Money / Zedd – Stay).

### Worst-First Table (18 rows; 5 more below cutoff with |error| 43–46 s)

| Mix time (mm:ss–mm:ss) | Ableton track (GT name + stem) | Slot | Axis | What the aligner did vs GT |
|---|---|---|---|---|
| 52:42–53:19 | Dog Days Are Over – Vocals Only (Florence + The Machine) · acappella | 36w1 | SET_START +84 s / REF_OFFSET −143 s | Placed 84 s too late (pred 54:06, GT 52:42) AND pred_ref is 0:00 while GT is at 2:23 — the aligner stacked it into the next section AND grabbed the song's intro instead of the late-chorus entry. Both axes wrong simultaneously. |
| 17:30–18:13 | Bingo Players – Cry (A-Trak & DiscoTech Remix) · regular | 11w4 | SET_START +45 s / REF_OFFSET −124 s | Placed 45 s late (pred 18:15, GT 17:30) and pred_ref 0.1 s vs GT 124 s — aligner landed it in the next Feel So Close / Calvin Harris block and started at the top of the reference instead of 2:04. |
| 43:34–43:54 | Lil Jon & The East Side Boyz – Get Low (Acapella) · acappella | 28w2 | SET_START −51 s / REF_OFFSET +109 s | Placed 51 s early (pred 42:43, GT 43:34); pred_ref 90 s vs GT −19 s (expected near the very start). The aligner displaced it into the Fall Out Boy window and grabbed a late-verse position in the ref. |
| 35:14–37:24 | Panic! At The Disco – I Write Sins Not Tragedies (Acapella) · acappella | 23w2 | SET_START +106 s | Placed 106 s too late (pred 37:14, GT 35:14). This is a 2-segment multiseg entry — GT starts at mix 35:14 but the aligner moved it to 37:14, right after the GT window closes. Segment structure not reproduced. |
| 51:03–52:10 | Jay Hardway – Bootcamp · regular | 35 | SET_START +100 s | Placed 100 s too late (pred 52:45, GT 51:03). Multiseg bed; the aligner pushed it into the Florence + The Machine / Dog Days block. |
| 58:15–58:36 | Steve Miller Band – The Joker (Vocals Only) · acappella | 39w5 | SET_START +43 s / REF_OFFSET −92 s | Placed 43 s late (pred 58:58, GT 58:15) and pred_ref 1.5 s vs GT 93.5 s — aligner missed the entry by one section and grabbed the song opening instead of the ~1:34 late-verse entry. |
| 16:50–18:50 | Feel So Close – Instrumental · instrumental | 11 | SET_START +90 s | Placed 90 s late (pred 18:20, GT 16:50). This is a 9-segment complex multiseg (repeating 7.5 s fragments) spanning 2 minutes; the aligner displaced it into the Bingo Players block and the segment decode is entirely wrong. |
| 39:50–41:12 | DJ Snake & Yellow Claw – Ocho Cinco (Henry Fong Mashup) · regular | 26 | SET_START +53 s / REF_OFFSET +90 s | Both axes broken: placed 53 s late (pred 40:43, GT 39:50) and pred_ref 165.9 s vs GT 75.9 s — wrong mix window and wrong song section within that window. |
| 35:58–36:42 | Two Friends – Out Of Love (Culture Code Remix) · regular | 22w3 | REF_OFFSET +89 s | Placed correctly in mix (off 1 s), but pred_ref 137.7 s vs GT 48.3 s — aligner jumped to the 2:18 position when the DJ played from the 0:48 entry. |
| 54:39–54:59 | The Fray – You Found Me (Vocals Only) · acappella | 37w2 | SET_START +89 s | Placed 89 s too late (pred 56:08, GT 54:39). Oddratio class (extreme tempo change in GT); the aligner cannot handle the stretch and displaced it to the next block. |
| 25:10–25:21 | Alan Walker – Faded (Official Acapella) · acappella | 15w3 | SET_START +86 s | Placed 86 s too late (pred 26:36, GT 25:10). This is a 2-segment multiseg; the aligner fell to the Deniz Koyu block and the segment structure is not reproduced. Note: identity also wrong (slot 15w3 in scorer = pred `rl7yhuf` is actually the correct Alan Walker acappella but misidentified vs GT). |
| 13:08–13:53 | Whethan – Savage (Studio Acapella) · acappella | 8w2 | SET_START +85 s | Placed 85 s too late (pred 14:33, GT 13:08); loop class. **Identity also wrong**: aligner put Whethan – Savage at 14:33 which overlaps the Galantis – No Money / Zedd – Stay window. The actual Whethan Savage acappella should be at 13:08 looping a 21.6 s segment twice. |
| 36:37–37:00 | Jason Mraz – I'm Yours (Vocals Only) · acappella | 23w1 | REF_OFFSET −78 s | Placed correctly (off 1.8 s), but pred_ref 0.7 s vs GT 78.9 s — aligner grabbed the intro when the DJ used the ~1:19 verse section. |
| 20:51–21:07 | Calvin Harris – Sweet Nothing (Official Acapella) · acappella | 12w3 | REF_OFFSET +75 s | Placed correctly (off 0.9 s), but pred_ref 105.2 s vs GT 30.1 s — aligner landed at ~1:45 when the DJ used the ~0:30 entry (post-intro chorus). |
| 15:42–16:51 | Zombie Nation – Kernkraft 400 (VAVO 2016 Bootleg) · regular | 10w3 | SET_START +15 s / REF_OFFSET −73 s | Placed 15 s late; pred_ref 34.5 s vs GT ~108 s (GT audible_start_s 955.6). **Gain-silenced intro**: GT notes audible_frac 0.80 with audible_start at 15:56, so the clip fades in slowly — the aligner should target 15:56 not 15:42, and the ref-offset error of −73 s means it grabbed a much earlier track section. |
| 42:10–43:28 | Fall Out Boy – Sugar We're Goin Down (Vocal Only) · acappella | 28w1 | SET_START +68 s | Placed 68 s too late (pred 43:18, GT 42:10). Oddratio class; the aligner displaced the acappella forward by one transition, colliding with the Lil Jon window. |
| 12:38–13:08 | Katy Perry – Dark Horse (Official Studio Acapella) · acappella | 8w1 | REF_OFFSET +66 s | Placed correctly (off 0.1 s), but pred_ref 107.8 s vs GT 46.2 s — aligner started at ~1:48 while the DJ entered at ~0:46 (beginning of the rap verse). |
| 17:58–18:13 | Daft Punk – Around The World (Acappella) · acappella | 11w5 | REF_OFFSET −65 s | Placed correctly (off 1.4 s), but pred_ref 16.8 s vs GT 64.3 s — aligner grabbed the first chorus when the GT is in a later chorus at ~1:04. |

*Below cutoff (|error| 43–63 s): 18w3 (DJ Snake & AlunaGeorge – You Know You Like It acap, −47 s SET_START), 9w2 (Tiësto – Red Lights acap, −46 s SET_START), 16w2 (Kelly Clarkson acap identity miss + 63 s SET_START), 38w2 (Maroon 5 Cold acap, +63 s SET_START), 1w3 (Carly Rae Jepsen – Call Me Maybe acap, −24 s SET_START / +50 s REF_OFFSET).*

---

## Cross-Set Notes

- **Slot 33w3 (BB12, −746 s)** is an extreme outlier driven by the Slide acappella appearing twice in the mix; the aligner picked the wrong instance. This single error accounts for a large fraction of the total BB12 set_start loss.
- **Ref-offset failures cluster at pred_ref ≈ 0**: in both sets the most common ref-offset failure is the aligner returning near-zero ref positions when the GT is 60–200 s in. This is the "grabbed the intro" pattern — HuBERT / chroma placement found the right mix window but the internal decode defaulted to the track's beginning.
- **Oddratio spans universally fail trajectory** (7 % traj-acc on BB11, 28 % fiber-aware on BB12). These are cases with unusual tempo ratios (< 0.9 or > 1.15); the Viterbi offset decoder cannot represent the warped path and either skips the span or extrapolates incorrectly.
- **Identity misses in overlapping windows** (slots 4/4w2/4w3 in BB12, 9w2/8w2 in BB11): when 3+ tracks play simultaneously in a mashup medley the MERT sequence head reliably picks the wrong neighbor. This contributes ~6 % of total GT-seconds lost.
- **Gain-silenced intro spans** (BB11 slot 10w3 / BB12 slot 6 area): GT marks audible_frac < 1.0 with an explicit audible_start_s; the scorer measures placement against the audible onset, but these spans still show large ref-offset errors because the clip's internal entry point is wrong.
