# Decision log

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 1.0  |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |

---

One entry per meaningful strategy decision, grouped by round, newest at the bottom. Routine
refits, housekeeping and cosmetic changes are omitted; those are in each notebook's change
log. Dates are as recorded in the notebooks — Function 8's change log is undated, so its
entries carry a round only.

**Outcome** is filled in after the result arrived, and is left as recorded even where the
decision turned out badly. Several entries below record decisions that were later reversed
or withdrawn; that is the point of keeping the log.

## Template

**Date:**
**Round:**
**Decision:**
**Why:**
**Outcome:**

---

## Rounds 1–4 — scheduled annealing

### F5 — Replace `log1p` with a signed-log transform

**Date:** 11/08/2026 · **Round:** 4
**Decision:** Replace `np.log1p(y)` with a signed-log transform throughout the pipeline, and
re-run the RBF/Matérn comparison afterwards.
**Why:** `log1p` is undefined for `y ≤ −1`, and the round 3 output was below −1. Every round
from that point onward would have produced NaNs.
**Outcome:** Pipeline fixed. This is a bug exposed by a result rather than a modelling choice,
and it is the only output transform applied to this function.

### F7 — Restrict acquisition search to a local neighbourhood

**Date:** 13/08/2026 · **Round:** 4
**Decision:** Stop searching the full unit cube; restrict acquisition optimisation to a
neighbourhood around the current best point.
**Why:** Three rounds of annealed β had put every proposal on a domain boundary and beaten
nothing in the initial data.
**Outcome:** Necessary but not sufficient — the boundary-seeking persisted inside the local
box too, which is what forced the round 5 change below.

---

## Rounds 5–7 — local trust regions

### F3 — Correct a corrupted round 3 output, and pivot to local search

**Date:** 12/08/2026 · **Round:** 5
**Decision:** Correct the round 3 output from 0.5572213082937378 to −0.05549717617041605 at
source, and switch from global candidate search to a local search bounded around the current
best point.
**Why:** Three of four queried points had clustered in the same corner region without beating
an incumbent supported by a single observation.
**Outcome:** Correct on both counts. The pivot produced the first record (−0.0325) in round 6.
The data correction later proved to be one instance of a wider off-by-one — see the final
entry in this log.

### F1 — Switch from global to bounded local EI

**Date:** 20/08/2026 · **Round:** 5
**Decision:** Restrict EI search to a window around the current best at (0.731, 0.733) rather
than continuing global exploration.
**Why:** Four rounds of global search had produced outputs between 1e-37 and 1e-102 and no
improvement.
**Outcome:** **Wrong, and expensively so.** This began seven rounds of refinement that moved
the best value not at all, because the problem was the model's scale, not the search region.
Diagnosed in round 11.

### F2 — Reverse the schedule: raise β and move to an under-sampled quadrant

**Date:** 20/08/2026 · **Round:** 5
**Decision:** Raise β from 1.2 to 1.8 — against the annealing schedule — and restrict the
search to an identified under-sampled quadrant.
**Why:** Four rounds in the same corridor had not improved on the supplied best, and round 4
had shown the corridor was not a flat plateau.
**Outcome:** Correct. Round 6 produced the first improvement at 0.619497.

### F6 — Pivot from global UCB to a trust region

**Date:** 20/08/2026 · **Round:** 5
**Decision:** Local trust-region search, radius 0.15, centred on the current best, β = 0.75.
**Why:** Global UCB kept re-deriving the same failed corner.
**Outcome:** **Failed in an informative way.** The round 5 query landed on the exact same
corner of the trust region as round 4, showing the box-boundary-seeking pathology persists
even locally. This motivated the round 8 switch away from UCB entirely.

### F7 — Constrain `x3` against the surrogate's own belief

**Date:** 20/08/2026 · **Round:** 5
**Decision:** Constrain `x3` to an empirically supported band, overriding the GP's assessment
that `x3` was irrelevant.
**Why:** The scan's top candidates differed from the incumbent almost entirely in `x3`. The
GP called that dimension flat; the observed data contradicted it.
**Outcome:** The largest weekly gain for this function to that point. Also, eight rounds
later, the one episode that would overturn a mistaken rule — see round 13.

### F4 — Add a compositional kernel

**Date:** 18/08/2026 · **Round:** 5
**Decision:** Add a broad-RBF-plus-narrow-RBF composite as a third kernel candidate.
**Why:** The dataset now contained an extreme value against an otherwise ordinary range, and
no single stationary kernel can represent a smooth background and a sharp local spike at once.
**Outcome:** Reasonable given what was believed at the time. The premise is now in question —
see the final entry.

### F5 — Abandon "bracket `x2` only, hold the rest tight"

**Date:** 18/08/2026 · **Round:** 5
**Decision:** Drop the assumption that `x1`, `x3` and `x4` are safe to hold near their
boundary values, and switch to a tight local box across **all four** dimensions.
**Why:** Round 4 moved `x3` from 0.999999 to 0.9557 with `x2` almost unchanged and lost about
370 in yield.
**Outcome:** Correct, and the start of a chain that eventually established all three of
`x1`/`x3`/`x4` as boundary-pinned.

### F6 — Manual override against the automated search

**Date:** 02/09/2026 · **Round:** 6
**Decision:** Override the acquisition function. Raise sugar and milk while keeping flour,
eggs and butter near the incumbent profile.
**Why:** Two rounds of automated search had produced nothing, and the hypothesis being tested
was one the search could not reach because the collected data contained no evidence for it.
**Outcome:** **y = −0.465434, an improvement of 0.245 and the best result to that point.** The
hypothesis the repeated automated search kept failing to find was correct.

### F4 — Trust-region search with a novelty constraint

**Date:** 25/08/2026 · **Round:** 6
**Decision:** Restrict acquisition optimisation to a small box around the round 3 point, with
an explicit novelty constraint so the query cannot re-confirm an already-known location.
**Why:** Round 4 had already landed 0.041 away and returned an ordinary value, disproving a
broad-basin hypothesis; the question was now whether the feature is extremely narrow.
**Outcome:** The trend rose with proximity initially, then flattened by round 8 rather than
climbing — the evidence that ended this line in round 9.

### F3 — Feature-aware search

**Date:** 09/09/2026 · **Round:** 7
**Decision:** Bound `compound_3` tightly to its empirically good band `[0.25, 0.50]` while
leaving `compound_1` and `compound_2` essentially unconstrained.
**Why:** The learned length-scales showed `compound_1`/`compound_2` pinned at their upper
bound (nearly irrelevant) while `compound_3` was dominant with a short length-scale.
**Outcome:** Reverted one round later — unconstraining two dimensions sent the query back to
corner regions that had already failed.

### F5 — Revert to a tight confirmatory box

**Date:** 09/09/2026 · **Round:** 7
**Decision:** Return to the tightest box used so far (δ = 0.05) around the round 2 point, β
lowered sharply to 0.4.
**Why:** A robustness pass over a genuinely different region had returned 68.5 and found
nothing better.
**Outcome:** Set the record of 1915.052 that stood until round 10.

### F7 — Stop widening, start bracketing

**Date:** 09/09/2026 · **Round:** 7
**Decision:** Fit a quadratic through three bracket points, place the `x3` peak at 0.348, set
the band to `[0.28, 0.36]`, and tighten the radius from 0.20 to 0.12.
**Why:** Widening had exhausted its usefulness; the peak was now bracketed rather than
unknown.
**Outcome:** The start of the measured ladder that produced five consecutive records.

---

## Rounds 8–12 — one factor at a time, and evidence-driven region control

### F8 — Raise β to 3.0 and freeze `x8`

**Round:** 5
**Decision:** Raise β from 0.7 to 3.0, freeze `x8` as a confirmed nuisance dimension, and
raise the noise floor to 1e-4. Spend the query probing 0.23 from the basin.
**Why:** Expected Improvement had collapsed to identically zero everywhere — the model
believed nothing in the space could beat the incumbent — and a low-β UCB would only re-sample
within 0.1 of it. That confidence was not trustworthy: σ at the incumbent was about 1e-4,
meaning the GP was interpolating 44 points rather than regularising them.
**Outcome:** Returned 0.065 **below** the incumbent. A loss in objective value that confirmed
the basin and validated the calibration fix in the same round. The `x8` freeze was wrong and
stood unchallenged for four rounds.

### F8 — Abandon multi-coordinate proposals

**Round:** 8
**Decision:** Move one coordinate per round. Step on `x3` alone, targeting the point of
maximum disagreement between the two best-fitting models.
**Why:** Four straight failures from multi-coordinate GP proposals, none of which could be
attributed to a single cause when they failed.
**Outcome:** Ended the drought at 9.976026 and produced a clean attribution at the same time.
Three of this function's five improvements came from this method.

### F6 — Replace UCB with edge-avoiding mean exploitation

**Date:** 16/09/2026 · **Round:** 8
**Decision:** Shrink the trust region from 0.15 to 0.07 and switch from UCB to pure mean
exploitation over a dense interior grid.
**Why:** To stop the acquisition optimiser repeatedly landing on trust-region edges, which
plain UCB structurally re-introduces.
**Outcome:** Fixed the boundary bias. Also overconfident — this round predicted −0.380680 and
received −0.640971 — which is what motivated the round 10 move to Expected Improvement.

### F3 — Revert from feature-aware to a tight local box

**Date:** 17/09/2026 · **Round:** 8
**Decision:** Return to a ±0.15 local search centred on the actual best point, keeping
`compound_1`/`compound_2` moderate rather than exploring corners.
**Why:** The unconstrained feature-aware search had sent the query back to corner regions that
had already failed five times.
**Outcome:** y = −0.0125, a near-threefold improvement and the function's first large gain.

### F4 — Shell-constrained probe

**Date:** 08/09/2026 · **Round:** 8
**Decision:** Restrict the search to a ring around the round 3 point: no farther than 0.008
from it, no closer than 0.003 to any already-observed point.
**Why:** Every query in rounds 4 to 7 had been at least 0.010 away. This guarantees a
genuinely new test, closer than anything tried.
**Outcome:** The trend **plateaued** rather than climbing — 0.541 at distance 0.010 against
0.561 at 0.0116. This is the evidence that ended the chase one round later.

### F8 — Spend a query on the oldest untested assumption

**Round:** 9
**Decision:** Hold all seven other coordinates fixed and move `x8` from 0.999999 to 0.000000,
rather than attempt another refinement.
**Why:** Refinement gains had collapsed below the model's own σ, so a refinement query would
have bought nothing. The `x8` freeze had stood for four rounds on an untested belief.
**Outcome:** **−4.50σ.** Output fell from 9.976026 to 9.956026 against a prediction of
9.975402 ± 0.00431. A four-round structural assumption refuted, and `x8`'s length-scale fell
from its pinned bound to 232.67. **The single most valuable query of the thirteen.**

### F4 — Abandon the chase; diversify

**Date:** 23/09/2026 · **Round:** 9
**Decision:** Stop hyper-local refinement around the round 3 point. Switch to a diversification
search at β = 2.5, explicitly excluding everything within 0.15 of it.
**Why:** The gain from round 8 (+0.036) was marginal and non-monotonic relative to round 5.
This is a flat plateau, not an approach to a peak.
**Outcome:** Returned −20.576. Cost a query and found nothing, but the diagnosis was right and
round 10's reversion to the best *confirmed* point followed from it.

### F2 — Prioritise robustness over predicted mean

**Date:** 23/09/2026 · **Round:** 9
**Decision:** Stop taking exploratory jumps of the previous size near the peak; use a tight
±0.05 box directly around the current best.
**Why:** Round 8's miss suggested the marginally higher predicted mean nearby was not worth
the variance.
**Outcome:** **y = 0.711209 — the project best for this function**, a +0.091 single-round gain,
and never beaten in the four remaining rounds.

### F7 — Continue the ladder rather than take the EI optimum

**Date:** 23/09/2026 · **Round:** 9
**Decision:** Step `x3` to 0.360 rather than submit the EI-optimal point.
**Why:** The single-variable method had just been validated — `x3` 0.300 to 0.330 with
everything else fixed gave a cleanly attributable +0.050179 — while the EI optimum again
drifted onto three box bounds.
**Outcome:** 1.698012, a third consecutive record, plus a controlled pair establishing **+0.588
attributable to `x4` alone**.

### F4 — Distinguish the best *confirmed* point from the unconfirmed one

**Date:** 30/09/2026 · **Round:** 10
**Decision:** Revert to local exploitation, centred on the round 8 confirmed point (0.577)
rather than on the unreproduced extreme value.
**Why:** Making the distinction explicit, so that the reportable result does not rest on a
single unreproduced observation.
**Outcome:** The right call methodologically, and the basis for the final result. The
unconfirmed value is now believed to belong to a different function entirely.

### F5 — Widen in response to calibration, against the schedule

**Date:** 30/09/2026 · **Round:** 10
**Decision:** Raise β from 0.3 to 0.5 and loosen the search bounds on all four dimensions,
rather than continuing to narrow.
**Why:** Leave-one-out coverage had drifted to 82.8%, below the 95% target and alongside round
8's 75.0%. The surrogate was overconfident across the dataset, not at one point.
**Outcome:** **y = 1915.964, a new best.** The true `x2` optimum sat a touch further out than
the round 9 box allowed. A calibration diagnostic directly produced a record.

### F6 — Choose the trust-region radius by Expected Improvement

**Date:** 30/09/2026 · **Round:** 10
**Decision:** Sweep the trust-region radius from 0.07 to 0.30 and compute actual EI at the best
candidate in each box, taking the radius where EI peaks.
**Why:** Pure mean exploitation gave no principled way to pick a radius, and had been
overconfident in round 8.
**Outcome:** EI peaked at radius 0.20 and was flat out to 0.30 — a genuine local mode, not an
artefact of box size. **y = −0.401802, a new best** and the first improvement in four rounds.

### F7 — Scale the stride from 0.03 to 0.12

**Date:** 30/09/2026 · **Round:** 10
**Decision:** Increase the single-variable step size fourfold while keeping the method
single-variable.
**Why:** The ladder showed no flattening — +0.050179 then +0.051124 per 0.03 step — and three
queries remained. Scaling converts an established trend into score.
**Outcome:** 1.869 then 1.873; the gain rate then fell 45-fold, which bracketed the peak and
triggered the round 12 change.

### F1 — Cluster-informed anchor

**Date:** 05/10/2026 · **Round:** 11
**Decision:** Anchor the EI search on the output-weighted centroid of a "promising" cluster
rather than on `argmax(Y)`, using a two-cluster split on `sign(y)` and `log10|y|`. Tighten the
window to ±0.004.
**Why:** All four diagonal neighbours of the incumbent within 0.014 were now positive. Four
consistently positive neighbours look like a genuine basin rather than an isolated spike.
**Outcome:** No improvement, and the result that broke the whole approach open — see the next
entry.

### F4 — Clustering-informed angular exploration

**Date:** 05/10/2026 · **Round:** 11
**Decision:** Stop radial tightening. Test a new angular direction at the edge of the
characterised shell, inside the hot cluster's region but outside the saturated inner ball.
**Why:** k-means on all 40 points confirmed a single hot region with a far higher median
output, validating continued focus — but tightening had been shown unproductive.
**Outcome:** No improvement, and it confirmed that the plateau extends in multiple directions
rather than sloping toward a peak.

### F1 — Correct the diagnosis and abandon local refinement

**Date:** 14/10/2026 · **Round:** 12
**Decision:** Abandon local EI refinement entirely, including the planned repeat of the best
point. Fit an isotropic Gaussian in **log-space** to the six basin points and query the fitted
centre, about 0.05 away.
**Why:** Round 11's point was the closest query yet to the best (0.0047 away) but only the
second-lowest basin output, and the GP had predicted 4.108e-06 — a 3.8e10× overshoot. Earlier
notes blaming noise or a rugged surface were **wrong**. Root cause: the GP is fitted on raw
`y`, whose spread of about 8e-4 is set by one −3.6e-03 reading and is about twelve orders of
magnitude larger than the basin's own spread of 3e-16. The basin is invisible to it. The
log-space model reproduces all six outputs to 2% and ranks them exactly by distance to the
fitted centre, which resolves the standing "closer to best is not higher" anomaly — the basin
is a **flank**, not a peak.
**Outcome:** **y = 4.743560e-07, about 6e8× the old best.** Eleven rounds of search had been
exploiting a model that could not see the target.

### F6 — Cap EI rather than follow it

**Date:** 14/10/2026 · **Round:** 12
**Decision:** Introduce a capped-EI variant with butter's candidate range explicitly bounded
below the level that had just failed.
**Why:** The standard EI sweep still pointed back toward the near-edge butter region at
0.996–0.998 that had already underperformed. The GP's length-scales are long enough that one
bad edge result barely dents its optimism there — it does not learn that lesson on its own.
**Outcome:** **y = −0.362411, a new best.** Lowering butter from the edge helped.

### F4 — Replace the global GP with a local linear surrogate

**Date:** 14/10/2026 · **Round:** 12
**Decision:** Fit a ridge regression to the seven observations within 0.06 of the best
confirmed point, validated by leave-one-out, and take a bounded uphill step of radius 0.025
using bootstrap-ensemble UCB. Also add a kernel tie-break rule: within 1% on leave-one-out
RMSE, choose by fixed priority rather than numerical noise.
**Why:** The global GP was dominated by the single extreme value, with leave-one-out RMSE
around 417 for every kernel — a statistical tie — making its suggestions unstable between
machines.
**Outcome:** **The step failed.** Predicted about 1.32, returned 0.534. Diagnosed one round
later as curvature in `x2`, which the straight-line model could not represent.

### F2 — Spend a free option on an evidence gap

**Date:** 14/10/2026 · **Round:** 12
**Decision:** Probe the genuinely unsampled `x2 ≈ 0.15–0.45` gap inside the proven `x1`
corridor, raising β from 0.25 to 1.5.
**Why:** The best (0.711209) is already locked in from round 9 and cannot be lost, so this
query carries no downside. A fourth tight perturbation next to a now-proven-unstable
neighbourhood would buy nothing for the final decision.
**Outcome:** Informative rather than improving. It is what made round 13's comparison possible.

### F5 — Spend the last free option on a full-domain probe

**Date:** 14/10/2026 · **Round:** 12
**Decision:** Open the search box to the full domain at β = 1.0 and let the acquisition
function choose.
**Why:** The corner had plateaued — the best refinement candidate offered an expected
improvement of about 0.3% — and round 12 was the last query where a miss costs nothing.
**Outcome:** Found a second region at about 1085–1089, real but lower than the corner, and
below the threshold set in advance for refining it.

### F7 — Stop stepping outward, convert the bracket

**Date:** 14/10/2026 · **Round:** 12
**Decision:** Step `x3` to 0.542 rather than continue outward.
**Why:** The gain rate had fallen from +0.0428 to +0.00096 per 0.03, a 45-fold drop, so the
peak was bracketed. Two independent methods agreed on its location to within 0.0008 — a
quadratic through the last three rungs gave 0.5428, the GP's single-dimension argmax gave
0.542. Predicted +0.0227 with the lowest uncertainty of any available move.
**Outcome:** 1.899000, a sixth record. The interpolated vertex beat both bracketing points,
confirming the peak by measurement.

---

## Round 13 — structural change

### F3 — Stop trusting the GP for two of three dimensions

**Date:** 24/10/2026 · **Round:** 13
**Decision:** Fit a quadratic to `y` along the PC1 ridge through the five consecutive queries
sharing `compound_3 ≈ 0.415–0.418`, take its peak for `compound_1`/`compound_2`, and let the
GP set `compound_3` only. β = 0.
**Why:** Matérn was retained on leave-one-out (RMSE 0.0749 against RBF's 0.0818) but its
`compound_1`/`compound_2` length-scales stay pinned at the upper bound of 10, so the posterior
is almost flat exactly where resolution is needed — about −0.0101 to −0.0103 at three nearby
points that actually scored −0.0040, −0.0101 and −0.0031. Its maximiser is a weak guide there.
β = 0 because no query follows, so an exploration bonus is a cost with no benefit.
**Outcome:** Returned −0.016509; no improvement. The best remains round 11's −0.003116.

### F2 — Admit indifference and break the tie without a model

**Date:** 24/10/2026 · **Round:** 13
**Decision:** Compare two candidates — refine the spike, or fill the largest empty `x2`
interval in the proven band — and choose by a model-free space-filling tie-break. No β, no UCB.
**Why:** All global surrogates give both candidates about a 6% chance of beating 0.7112, and
the empirical semivariance shows nearby points do not make an outcome more predictable. The
data are genuinely indifferent between them, and manufacturing a preference would be dishonest.
**Outcome:** 0.649151 — above the forecast mean of 0.583 ± 0.085, below the best. Record stands
at round 9.

### F4 — Local ARD GP, and a probe rule fixed in advance

**Date:** 24/10/2026 · **Round:** 13
**Decision:** Replace the linear model with a local ARD GP fitted only to the eight plateau
points. Keep its step as a benchmark but submit an **exploration probe** instead: 0.05–0.12
from the best confirmed point, at least 0.05 from every queried point, scored by mean + 2σ.
**Why:** The round 12 failure was curvature — `y` peaks in `x2` near the round 8 value, so
moving `x2` down costs value while moving `x4` down still helps. The local ARD GP's
leave-one-out RMSE is **0.033** against 0.54 for a flat plateau and 0.36 for the best ridge,
and it predicts the held-out rounds 4, 11 and 12 to within 0.02–0.03. Plateau refinement had
stopped paying, so the only move that could change the outcome was exploration — and a miss
costs nothing.
**Outcome:** Forecast 0.364 ± 0.386, observed **0.591 — a new confirmed best.**

### F5 — Probe the unexplored far vertex

**Date:** 27/10/2026 · **Round:** 13
**Decision:** Query the vertex `(0.999999)⁴` as a hypothesis test for a second, higher maximum,
with a `FINAL_MODE` switch retaining the bounded corner-ramp step as an alternative.
**Why:** Every query so far sits in the corner basin or the interior; none at the far corner
where `x2`, `x3` and `x4` are simultaneously maximal. The GP assigns most unexplored vertices
the same maximum uncertainty (σ ≈ 2.05 in log-space), so the model alone cannot rank them. This
is a hypothesis test, not an optimisation.
**Outcome:** **y = 8662.405001 — about 4.5× the previous best**, against a GP prediction of
roughly 124. For reference the corner ramp extrapolates to only about 2720 at `x2 = 1`, so `x1`
matters a great deal. 99.999% of the reference best.

### F6 — Fix a corrupted row, then hedge across an ensemble

**Date:** 27/10/2026 · **Round:** 13
**Decision:** First, correct the data: the round 11 entry was a copy of round 1's point and
output. Freeze the round 12 section to the data as of that decision and re-run it on corrected
data for transparency. Then replace the surrogate with a standardised Matérn 5/2 ARD GP used
inside a **30-member bootstrap ensemble**, and choose by bag-averaged Expected Improvement.
**Why:** Refitting the usual RBF GP on the corrected 32 points *still* pushes butter back to
about 0.997 — the same edge that gave −0.520 — while the local evidence points the other way:
butter at 0.9996, 0.981 and 0.939 gave −0.520, −0.402 and −0.362. The Matérn fit has lower
leave-one-out error (0.227 against 0.324) and lower forward error (0.128 against 0.202) but its
stated uncertainty is too narrow, with forward z reaching −4.0 — so it is better *and* not to
be trusted alone.
**Outcome:** **y = −0.213554, a new best**, beating round 12 and also beating the ensemble's own
predicted mean of −0.342.

### F7 — Derive a rule, test it, and withdraw it

**Date:** 28/10/2026 · **Round:** 13
**Decision:** Withdraw the "move type" bias rule that had treated multi-coordinate moves as
unreliable since round 8. Submit the GP optimum with `x1` held in its proven band — a
six-coordinate move.
**Why:** All three queries the rule rested on also carried a boundary or sub-floor coordinate,
a cause already established in round 8, so the two explanations were perfectly confounded in a
sample of three. And of the two multi-coordinate queries with no boundary value, one was round
5 — the largest weekly gain to that point — which the pooled statistic had concealed. Keeping
the boundary lesson without the unsupported rule reverses the final decision.
**Outcome:** **y = 2.240937, the project best for this function and a +0.342 gain — the largest
single round of the whole project.** The model under-predicted by 0.225.

### F1 — Refit globally as a two-source model and constrain the EI rule

**Date:** 21/10/2026 · **Round:** 13
**Decision:** Replace the local log-Gaussian with a **global two-source log-magnitude model**
fitted to all points, and choose the query by maximising Expected Improvement along the ray
toward the stronger source, subject to **P(beat best) ≥ 50%**, with the unknown sign-flip radius
modelled explicitly by Monte Carlo.
**Why:** The round 12 local model under-predicted by about 650× while the global two-source
model was within about 6×. It places a much stronger source near (0.628, 0.629) — the basin was
a flank of it. The constrained rule was preferred to unconstrained EI, which offered only about
a 22% chance of beating the incumbent.
**Outcome:** Predicted 2.56e-05, returned **4.240382e-05** — about 89× the round 12 best and
about 12× the reference best. A ratio of 0.60, the best out-of-sample prediction of any model
used in this project. **Strategy confirmed, not changed.**

### F8 — Solve the function in closed form

**Round:** 13
**Decision:** Stop optimising. Identify the function as **additively separable quadratic in 17
parameters**, fit it exactly, validate, and submit its analytic argmax.
**Why:** Twelve rounds of kernel, noise-floor and acquisition tuning had run inside a model
family that could not represent the target. A stationary kernel with a long length-scale
represents a weak additive effect as no effect at all.
**Outcome:** The recovered model reproduces all 52 prior observations and holds out-of-sample
to residuals ≤ 1e-15. Predicted 10.0000000000; returned **exactly 10.0** — the global optimum,
matching the reference best. Seventeen observations suffice to identify the function and forty
were supplied before the first query: **the answer was over-determined before the first query
was ever submitted.**

---

## Open items

### Unresolved round-3 attribution between Functions 4 and 5

**Date:** 01/11/2026 · **Round:** post-submission audit
**Decision:** Record the discrepancy rather than silently correct it; treat the portal files as
authoritative for reported convergence while leaving each notebook on the value it was built on,
so historical cells still reproduce.
**Why:** In `data/accumulated/week3/outputs.txt`, six of eight values match their notebook
exactly. Two are shifted by one position against each other: 2602.253793 sits in Function 5's
slot but was used by Function 4, and −1.359780 sits in Function 6's slot but was used by both
Function 5 and Function 6. Two functions cannot own the same observation. The portal ordering is
the likelier reading on range grounds, and Function 3 independently found and corrected this
same off-by-one in round 5.
**Outcome:** **No final best is affected** — Function 4's record and Function 5's were both set
in round 13. Two intermediate narratives inherit the ambiguity: Function 4's chase of an extreme
value across rounds 4 to 9, and Function 5's signed-log transform introduced to handle a value
below −1. Both are flagged wherever they appear. Pending resolution with the course platform.
