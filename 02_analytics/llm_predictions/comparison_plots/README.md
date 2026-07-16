# LLM vs. human comparison — alternative views

Same underlying data as `../result_vwp50_scene_table.csv` (GPT-2, 50 items)
and `../../Analysis_pipeline_m/Output/fixation_proportions.csv` /
`growth_curves.csv` (human gaze, 5 pilot subjects), presented a few different
ways. All built from real data — nothing simulated.

## Files

- **`combined_timecourse_50items.png`** — time-course view (like the original
  Altmann & Kamide Fig. 2 style): human fixation proportion over 50ms bins
  from verb offset, target vs. mean distractor, both conditions, with GPT-2's
  item-averaged P(target)/P(distractor) as flat reference lines. Best for:
  showing the anticipatory-looking *shape* over time, and how far human data
  currently sits from GPT-2's confidence.

- **`1_scatter_item_agreement.png`** — one dot per (item × condition) = 99
  points (50 items × 2 conditions, minus 1 missing cell — item 35 has no
  restrictive-condition trials in the current data). X = GPT-2 P(target),
  Y = human mean fixation proportion on target. Dashed line = perfect
  agreement. Spearman ρ printed in the title. Best for: directly testing the
  research question ("are LLM probabilities similarly distributed to human
  probabilities?") at the item level — this is the plot to lead with if
  asked for a single correlation number.

- **`2_bar_headline_comparison.png`** — four bars: GPT-2 vs. human "target
  advantage" (target − mean distractor), by condition, pooled across all 50
  items. Best for: a single slide takeaway number.

- **`3_dotplot_all_items_ranked.png`** — Cleveland dot plot, all 50 items
  (restrictive condition), sorted by GPT-2 confidence, with the matching
  human value plotted alongside each one. Best for: spotting whether *any*
  individual items behave as GPT-2 predicts (a few do — items 6, 7, 23, 31,
  37 — most don't), rather than only looking at the pooled average.

- **`item_level_llm_vs_human.csv`** — the merged per-item table these plots
  are built from (Item, condition, GPT-2 P_norm, human fixation proportion,
  n trials).

## Headline numbers (see script for exact computation)

- Spearman ρ (item-level, both conditions pooled) ≈ 0.14, p ≈ 0.16 — not a
  significant relationship yet.
- Target advantage: GPT-2 restrictive = 0.71, human restrictive = −0.02.
  GPT-2 non-restrictive = 0.10, human non-restrictive = −0.05.

Bottom line across all four views: GPT-2 is confidently graded (spans
0 → 1 across items, and separates the two conditions clearly); human
fixation data is close to floor almost everywhere, for the reasons already
documented in `02_analytics/llm_predictions/README.md` and the QC slides
(critical window mostly falls on the subject figure, not the objects; small
pilot n). These plots don't change that diagnosis — they just show it from
different angles for whichever one reads best in a given context.

- **`4_scatter_expandedAOI.png`** — same "LLM probability vs. human
  probability" scatter, but with the AOI **expanded**: besides the original
  object box, gaze samples in the corridor between the subject figure and
  each object (up to 150px either side of the subject→object line) now also
  count toward that object, instead of being thrown into "elsewhere". This
  drops the no-object-gaze rate from 79% → 56% — still over half of
  critical-window samples never leave the subject figure, so most points are
  still pinned near x=0. Item-level correlation barely moves (ρ=0.14 → 0.14,
  still n.s.).

- **`5_topchoice_comparison.png`** — the one genuinely encouraging result:
  instead of comparing raw probabilities, ask a coarser, more robust
  question — *among trials with at least one on-object sample, is the
  target the most-looked-at object?* Restrictive: LLM picks target 85% of
  the time, humans 44% (chance = 25%, so clearly above chance and clearly
  higher than non-restrictive at 29%, right where chance sits). This
  argmax-style question survives the sparse-sample problem much better than
  raw proportions do, because it only conditions on trials that have a
  usable answer instead of averaging in every trial as a zero.

- `item_level_human_expandedAOI.csv`, `item_level_llm_vs_human_expandedAOI.csv`
  — supporting data for the two plots above.

- **`6_dumbbell_topchoice.png`** — same restrictive/non-restrictive ×
  LLM/human numbers as plot 5, redrawn as a dumbbell (two dots + connecting
  line per condition) with a 95% Wilson CI on the human point. This is the
  one that actually settles the "is this real" question: restrictive's CI
  (31–57%) excludes chance; non-restrictive's CI (18–43%) does not.

- **`7_waffle_topchoice.png`** — the same four numbers as 100-square icon
  grids. Purely a presentation-register alternative to the bar/dumbbell
  forms above — not more rigorous, just more immediately legible for a
  lay audience.

- **`8_deviation_from_chance.png`** — the same four numbers reframed as
  percentage points above/below the 25% chance baseline (LLM restrictive
  +60pp, human restrictive +19pp, LLM non-restrictive +13pp, human
  non-restrictive +4pp). Answers "how big is the effect" more directly than
  a raw percentage does.

- `item_level_human_angularAOI.csv` — human proportions recomputed with the
  angular-sector AOI (see table above).

## Three AOI definitions, compared

| AOI method | no_object_gaze rate | restrictive: human top-choice=target (n scored) |
|---|---|---|
| Original (pipeline default, object box only) | 79.4% | 44% (n=55) — via original AOI, see plot 5 |
| Corridor (object box + 150px-wide subject→object strip) | 56.3% | 44% (n=55) |
| **Angular sector** (unbounded wedge from subject center, nearest-angle wins — no "elsewhere" at all except inside the subject box) | **28.6%** | 43% (**n=90**) |

The angular method (`make_angular_aoi.py`) is the more defensible of the two
expansions: the corridor's 150px half-width is an arbitrary parameter with
no principled justification, whereas the angular method classifies gaze by
*direction from the subject* with no free width parameter — every pixel
outside the subject box belongs to whichever object is angularly closest.
It also nearly doubles the number of scoreable trials (55 → 90 for
restrictive) without changing the headline number: target is still the top
choice ~43-44% of the time in the restrictive condition, now on a much
larger base. Non-restrictive ticks up slightly (29% → 33%) but stays close
to the 25% chance line either way.

- **`9_aoi_method_comparison.png`** — the three-method table above, as two
  panels: coverage (no_object_gaze rate, dropping sharply left→right) and
  the headline finding itself (target = top choice, restrictive condition),
  which stays flat at ~43-44% across all three methods while its CI narrows
  as n grows. Use this when the point is "the AOI choice doesn't change the
  conclusion, it just changes how confident we can be in it."

- **`10_first_saccade_direction.png`** — a different, stricter question:
  not "which object gets the most total dwell time" but "where does gaze go
  the very first time it leaves the subject region." Restrictive = 31%
  (n=90), non-restrictive = 28% (n=85) — both close to chance, and the
  gap between conditions nearly disappears. **This is a genuinely weaker
  result than the top-choice/dwell-time metric** and is worth reporting
  honestly rather than only showing the flattering one: the first
  orienting movement looks close to non-diagnostic in this pilot data;
  it's sustained looking over the rest of the critical window (captured by
  the top-choice metric) that shows the restrictive-vs-non-restrictive
  separation. "First saccade toward target" here = first raw sample
  classified outside the subject box via the angular method, not a
  velocity-based saccade detector — a coarse proxy, not a formal
  saccade-detection algorithm.

- `first_saccade_by_trial.csv` — per-trial data behind plot 10.

- **`11_scatter_angularAOI.png`** — the item-level LLM-vs-human scatter
  (same form as plot 1) redone with the angular-AOI human values instead of
  the original AOI. Human values are no longer floored at 0 for most
  points — visibly more spread, several items now sit at 0.3–0.75 — but
  Spearman ρ is still weak and not significant (0.13, p=0.19). Worth
  showing alongside plot 1: better AOI coverage changes the *shape* of the
  data a lot without yet producing a reliable item-level correlation.

- **`12_verbonset_vs_verboffset_window.png`** — same top-choice metric
  (angular AOI), but the critical window is redefined as **verb onset →
  target onset** (mean 621ms) instead of the standard **verb offset →
  target onset** (mean ~250-350ms). Widening the window backward to
  include the verb's own articulation time gives much better coverage
  (no_object_gaze drops to 18%, n grows to ~100-102) but the effect
  *weakens*: restrictive 43%→31%, non-restrictive 33%→21% (now *below*
  chance). This is expected and actually supports the standard verb-offset
  definition: before the verb finishes being said, the listener doesn't
  yet have the word identity needed to disambiguate, so including that
  time just dilutes the signal with pre-disambiguation gaze rather than
  adding useful information. More coverage isn't automatically better if
  it comes from a theoretically weaker window.

- `verbonset_window_by_trial.csv` — per-trial data behind plot 12.

- **`13_aoi_expansion_illustration.png`** — not a results plot, a
  **method illustration**: the real item 1 stimulus image with all three
  AOI definitions drawn directly on it, side by side. Screen coverage
  (fraction of pixels assigned to *some* AOI, subject included):
  Original 34% → Corridor 54% → Angular 100%. Makes concrete what the
  numbers in the table above actually mean geometrically — the angular
  method has no gaps at all; every pixel outside the subject box belongs
  to whichever object is in that direction.

- **`14_scatter_corridorAOI.png`** — the corridor-AOI item-level scatter,
  redrawn in the same axis convention as plots 1 and 11 (x = GPT-2, y =
  human) so all three AOI methods are now directly comparable side by side.
  (Plot 4 has the same underlying data but on swapped axes, matching a
  teammate's earlier slide layout — kept as-is rather than edited, use 14
  for apples-to-apples comparison with 1/11.) ρ = 0.14, p = 0.17 — same as
  plot 4's numbers, just re-oriented.

## ⚠ Angular AOI v1 → v2: the wedges were unequal, now fixed

The original angular method (`make_angular_aoi.py`, feeding plots 5, 9, 10,
11, 12) classified by *nearest angle with no bound* — but the 4 objects
only span 150° of the 360° circle (15°/65°/115°/165°), so the two edge
objects (cake, ball) absorbed the entire empty 210° arc behind the subject,
making their wedges ~3x bigger than the two middle objects' (toy car, toy
train). `make_angular_v2_and_scatters.py` fixes this: every wedge is capped
at ±25° (the true local half-gap), so all 4 are equal-sized; the leftover
empty arc goes back to "elsewhere" instead of being force-assigned. See
`13b_angular_AOI_v2_fixed.png` vs. `13_aoi_expansion_illustration.png`
(panel 3) for the visual difference.

**Consequence: no_object_gaze rate goes back up (28.6% → 53.9%)** since
v1's 100% coverage was partly an artifact of the unfair wedges. **Plots 5,
9, 10, 11, and 12 still use the old unequal-wedge (v1) angular AOI and
have not been regenerated with v2** — treat their exact numbers as
superseded pending a rebuild; the qualitative story (dwell-time/top-choice
metric shows a real restrictive effect, first-saccade direction doesn't,
widening the window backward weakens everything) is unlikely to reverse,
but the precise percentages should be re-derived from v2 before using them
in anything final.

- **`13b_angular_AOI_v2_fixed.png`** — corrected illustration, item 1,
  showing the equal ±25° wedges (compare to plot 13's third panel).

- **`15_scatter_corridorAOI_humanX.png`**, **`16_scatter_angularAOI_v2_humanX.png`**
  — plots 4/14 and 11 redrawn with **human on the x-axis** (matching the
  teammate's original slide convention) instead of GPT-2. Same underlying
  numbers as 14 (corridor) and a fresh v2 computation (angular): ρ=0.14 and
  ρ=0.09 respectively — both still weak/non-significant. `item_level_human_angularAOI_v2.csv`
  is the corrected per-item human data.

## Final, consistent versions (use these two if you only show one illustration + one scatter)

- **`17_aoi_expansion_illustration_v2.png`** — the definitive AOI illustration:
  all three methods on item 1, side by side, with the angular panel now using
  the fixed **v2 equal ±25° wedges** (supersedes `13_aoi_expansion_illustration.png`'s
  third panel and `13b_angular_AOI_v2_fixed.png`, kept for history). Each panel
  is annotated with its free parameter (none / 150px corridor / ±25° wedge)
  and its per-pixel screen coverage: Original 34% → Corridor 54% → Angular v2
  85%. (This coverage number is *pixel-area* coverage of the image, not the
  same thing as the trial-level no_object_gaze rate in the table above, which
  is measured from actual gaze samples — the two aren't directly comparable,
  they answer different questions: "how much of the picture is AOI" vs. "how
  much of real gaze lands in some AOI".)

- **`18_scatter_all3methods_combined.png`** — all three item-level scatters
  (Original / Corridor / Angular v2) in one figure, same axes and scale,
  human on the x-axis throughout. ρ = 0.14 (original), 0.14 (corridor), 0.09
  (angular v2) — all still weak and non-significant (p > 0.16 in every
  panel). The visual story is identical across methods: LLM confidence
  spans the full 0–1 range and separates conditions cleanly, human data is
  floored near 0 for the large majority of items regardless of which AOI is
  used to measure it — widening the AOI changes which points move off the
  floor, but not the overall shape or the (lack of) correlation.

## v3: real dwell-time (ms) + a subject box that matches the actual figure

Two changes bundled together, both aimed at making the AOI more defensible
rather than just bigger:

**1. Dwell time, not sample counts.** All earlier metrics used *proportion of
gaze samples* in each AOI as a stand-in for dwell time, which only equals
real dwell time if sampling is perfectly uniform. It mostly is (~6.7ms,
~150Hz) but blinks and tracker dropouts create gaps. `make_v3_smallbox_dwelltime.py`
computes actual per-sample duration from consecutive `TIME` values (capped at
20ms so a dropout gap isn't misread as 20ms of "dwell" on whatever AOI the
last valid sample happened to be in), sums to milliseconds per AOI per
trial, and redefines top-choice as *target has the most dwell-ms among the
4 objects* instead of *target has the most samples*.

**2. Subject box shrunk to match the real figure.** The subject box
(`SUBJECT_HALF`) was never measured — it was `(420, 300)` half-extents
(840×600px) by inheritance from the pipeline default. Measuring the actual
boy silhouette directly off `1_pos1_sub.png` (background-subtraction, see
script) gives a tight bbox of **177×575px**, centered almost exactly on the
assumed `SUBJECT_CENTER`. The height half-extent (300) was already close to
correct (287.5 measured); the **width half-extent (420) was ~4.7× too wide**.
New box: half-width 150 (measured 88.5 + ~60px margin for fixation
imprecision), half-height unchanged at 300.

- **`19_subjectbox_oldvsnew_illustration.png`** — old vs. new subject box,
  all 3 AOI methods, item 1. The old box visibly swallows empty white space
  on both sides of the boy; the new one hugs the figure.

- **`20_dwelltime_topchoice_oldvsnewbox.png`**, **`v3_smallbox_dwelltime_summary.csv`**
  — headline number, angular AOI v2, dwell-time top-choice:

  | subject box | restrictive | non-restrictive |
  |---|---|---|
  | old (840×600) | 44% (n=57), 95% CI [32%, 57%] | 36% (n=56), 95% CI [25%, 49%] |
  | **new (300×600)** | **38% (n=72), 95% CI [27%, 49%]** | **32% (n=76), 95% CI [22%, 43%]** |

  The smaller, more accurate box recovers ~15 more scoreable trials per
  condition (fewer samples get "stuck" on the oversized subject region) but
  **the restrictive number drops from 44% to 38%**, and its CI now
  **overlaps** with non-restrictive's CI. Restrictive is still nominally
  above the 25% chance line, but with this stricter setup the two
  conditions are no longer clearly distinguishable from each other at this
  sample size — a more honest, less flattering picture than the old-box
  version. This is likely because the old, oversized subject box was
  artificially suppressing "no object gaze" trials in a way that happened
  to favor the restrictive condition's headline number; it wasn't a free
  improvement, just a different (and probably less defensible) bias.

- **`21_scatter_dwelltime_newbox_all3methods.png`** — item-level scatter,
  all 3 AOI methods, dwell-time-weighted, new subject box, human on x-axis.
  ρ = 0.08 / 0.16 / 0.04 (original / corridor / angular v2) — still weak,
  still not significant (p > 0.12 everywhere). Points cluster more toward 0
  and 1 than the sample-proportion scatters (18) because dwell-time
  proportion is conditioned on trials that have *any* object dwell at all,
  which is a smaller, more selected subset.

- `item_level_human_{original,corridor,angular_v2}_v3_smallbox_dwell.csv` —
  per-item dwell-time-weighted human data behind plot 21, new subject box.

**Bottom line:** neither change (dwell-time weighting, smaller subject box)
strengthens the LLM-human relationship — if anything the more defensible
setup makes the human-side effect *harder* to distinguish from chance, not
easier. This is consistent with the diagnosis throughout this README: the
limiting factor is the short critical window + n=5 pilot subjects, not the
AOI geometry or the counting method.

## Fine-grained time bins + cluster-based permutation test (Altmann & Kamide style)

Everything above collapses the whole verb-offset→target-onset window into one
number per trial. `make_growth_curve_cluster_test.py` instead does what the
original 1999 paper actually did for its statistical claim: slice the window
into small time bins (50ms) aligned to verb offset, and test bin-by-bin for a
restrictive-vs-non-restrictive difference — using angular AOI v2 + the
measured (smaller) subject box from the v3 section above.

**Method**: for each subject × condition × bin, average the trial-level
"proportion of samples on target" (of ALL samples, on/off object, matching
the pooled time-course convention). Only bins where **all 5 subjects** have
data in **both** conditions are tested (bins 0–350ms qualify; window length
ranges 210–945ms, so coverage drops off fast past ~250ms — see the coverage
bar under the chart). Paired difference (restrictive − non-restrictive) per
subject per bin → one-sample t-test per bin → adjacent significant bins
(uncorrected p<0.05, same sign) grouped into a cluster, cluster mass = sum
of t-values. Null distribution built by flipping each subject's condition
labels across *all* bins simultaneously (the correct exchangeability unit)
and recomputing max cluster mass — with **n=5 subjects there are only 2⁵=32
possible label flips, so the smallest achievable p-value is 1/32 ≈ 0.031**,
much coarser than the thousands of permutations a real-N study would use.

- **`22_growth_curve_finebin_clustertest.png`** — the curve: restrictive
  visibly separates from non-restrictive starting around 150–175ms after
  verb offset (13% vs. 7% by bin 4) and stays separated through the covered
  range, which is exactly the qualitative shape the original paradigm
  predicts. **No cluster reaches significance** at this sample size
  (`growth_curve_cluster_test_results.csv` is empty — no candidate cluster
  even crossed the uncorrected p<0.05 threshold). This isn't a null result
  so much as an underpowered one: with only 5 subjects and a 32-permutation
  floor, the test needs a much larger, more consistent effect than what
  pilot data of this size can supply, even though the curve shape itself
  looks like the textbook anticipatory-looking pattern.

- `growth_curve_finebin_by_subject.csv`, `growth_curve_finebin_pooled.csv` —
  per-subject and pooled bin-level data behind the chart.

## Regenerating

- `make_combined_timecourse.py` → `combined_timecourse_50items.png`
- `make_alt_views.py` → plots 1–3 + `item_level_llm_vs_human.csv`
- `make_expanded_aoi.py` → plots 4–5 + the two `*_expandedAOI.csv` files.
  Reclassifies every raw sample in
  `../../Analysis_pipeline_m/Output/raw_gaze_critical_window.csv` against an
  expanded AOI (object box + subject→object corridor, corridor half-width
  150px in image-pixel space) rather than reusing the pipeline's original
  `aoi` column.

All are plain pandas + matplotlib (plus `scipy.stats.spearmanr`), reading
`../result_vwp50_scene_table.csv` and files in
`../../Analysis_pipeline_m/Output/`. Run from anywhere; paths are absolute.
