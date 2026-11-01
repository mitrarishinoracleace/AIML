# Strategy

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 1.0 |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |

---

One narrative per function, covering all thirteen rounds: what was believed at the start,
what the data contradicted, what replaced it, and what it cost. For the pipeline that
produced each query see [`methodology.md`](methodology.md); for the dated record of changes
see [`decision_log.md`](decision_log.md).

---

## The arc common to all eight

Every function began with the same assumption — that the exploration–exploitation trade-off
is a dial to be wound down on a schedule — and every function ended up somewhere else. The
shared sequence is:

| Phase | Rounds | What it assumed | What the data said |
|---|---|---|---|
| Scheduled annealing | 1–4 | A high β buys useful exploration | At this sample size it buys domain boundaries and corners |
| Local trust regions | 5–7 | Narrowing around the incumbent converts confidence into score | It does, until the incumbent's neighbourhood is exhausted |
| One factor at a time | 8–12 | Interpretability is worth more than coverage | True, and it is also what prevented a wider search |
| Structural change | 12–13 | The model class itself is a decision variable | Six of eight final records came from this phase |

What transfers is narrower than "use Bayesian optimisation": **exploration is not only about
where to sample, but about which model of the function to trust.**

---

## Function 1 — Contamination source detection (2D)

**Result:** 7.710875e-16 → **4.240382e-05** (round 13), about 5.5e10×, and roughly 12× the
reference best of 3.50e-06.

**Rounds 1–4.** GP-UCB then Expected Improvement over the full domain, with an exploration
floor wound 3.0 → 2.0 → 1.5 and an explore:exploit split moving 90:10 → 60:40. Outputs
ranged from 1e-37 to 1e-102. Nothing beat the supplied best.

**Rounds 5–11.** Local EI refinement around the incumbent at (0.731, 0.733), with the window
tightening from ±0.15 to ±0.005 and the split reaching 2:98. Two manual overrides were spent
on proximity tests. Round 11 added a cluster-informed anchor — a two-cluster split on
`sign(y)` and `log10|y|`, with the search centred on the output-weighted centroid of the
promising cluster rather than on `argmax(Y)`. **Seven rounds of careful local refinement
moved the best value not at all.**

**The diagnosis, round 11.** The query closest yet to the incumbent (0.0047 away) returned
only the second-lowest basin output, and the GP had predicted 4.108e-06 — a 3.8e10×
overshoot. Earlier notes had blamed noise or a rugged surface. That was wrong. The GP was
fitted on raw `y`, whose spread of about 8e-4 was set by a single −3.6e-03 reading, roughly
twelve orders of magnitude wider than the basin's own spread of 3e-16. **The basin was not
modelled badly; it was numerically invisible.**

**Rounds 12–13.** A log-space Gaussian fitted to the six basin points reproduced all six to
2% and ranked them exactly by distance to a fitted centre — which resolved the standing
anomaly that points closer to the best did not score higher. The basin was a *flank*, not a
peak, with the best point sitting about 5σ from the implied centre. Round 12 queried that
centre and returned 4.74e-07, roughly 6e8× the old best. Round 13 refit globally as a
two-source model, which had predicted round 12 to within a factor of 6 where the local model
missed by 650×, and queried along the ray toward the stronger source under the rule *maximise
EI subject to P(beat best) ≥ 50%*. It predicted 2.56e-05 and returned 4.24e-05.

**What this function taught.** Re-modelling beat re-sampling. The improvement came from no
new observations at all — the same eleven points, read on a different scale. The early
exploratory queries that looked wasted turned out to be load-bearing: in log-space they pin
down the envelope width and establish that a second source exists.

---

## Function 2 — Log-likelihood of a mystery model (2D)

**Result:** 0.611205 → **0.711209** (round 9), +16.4%, and +6.4% on the reference best.

**Rounds 1–4.** UCB with β wound from 2.5 downward across the full domain. Round 4 landed
beside the rank-1 initial point and scored 0.436 against its 0.611, just 0.07 away in `x2` —
so the `x1 ≈ 0.7` corridor was good but not a flat plateau. Four rounds, no improvement.

**Round 5.** A deliberate reversal: β raised 1.2 → 1.8 and the search restricted to an
identified under-sampled quadrant, rather than continuing to lower β in the same corridor.

**Rounds 6–9.** Targeting untested gaps inside the confirmed corridor. Round 6 produced the
first improvement (0.619497). Round 9 tightened to a ±0.05 box directly around the incumbent
after a round 8 miss, prioritising robustness over a marginally higher predicted mean — and
returned **0.711209**, the project best for this function and a +0.091 single-round gain.

**Rounds 10–12.** Pushing toward pure exploitation produced two misses. Round 11's −5.7σ miss
forced a refit in which the fitted noise rose sevenfold and the `x2` length-scale saturated.
A β sweep from 0.15 to 2.0 showed the suggested point barely moved below β ≈ 1, establishing
that **the corrected kernel, not β, was what fixed the overconfidence**. Round 12 spent a
free option on a genuinely unsampled gap at `x2 ≈ 0.15–0.45` inside the proven corridor.

**Round 13.** Two candidates were compared under the project's own surrogates: refine the
spike, or fill the largest empty `x2` interval in the proven band. All global surrogates gave
both about a 6% chance of beating 0.7112, and the empirical semivariance showed nearby points
did not make an outcome more predictable. With the models genuinely indifferent, the choice
was made by a model-free space-filling tie-break. It returned 0.649151 — above the forecast
mean, below the best, so the record stands at round 9.

**What this function taught.** Exploitation is only as good as the model's confidence, and
that confidence was inflated by a near-zero noise estimate. It is also the one function where
the honest final answer was "my models cannot rank these two options," and saying so was
better than manufacturing a preference.

---

## Function 3 — Drug compound mix, minimising adverse reactions (3D)

**Result:** −0.034835 → **−0.003116** (round 11), about 11× smaller penalty, and roughly 2.3×
better than the reference best.

**Rounds 1–4.** GP-UCB at β = 2.5. The corner queries of rounds 1, 3 and 4 scored between
−0.050 and −0.082, and the near-origin query in round 2 returned −0.173, the worst value the
function ever recorded. With 15 to 18 points in three dimensions the GP's uncertainty was
large everywhere, so a high β simply sent the search to the edges of the cube. **The failures
taught more about where not to look than about where the optimum was.**

**Round 5.** A critical data correction — the round 3 output had been recorded as
0.5572213082937378 when it should have been −0.05549717617041605 — and a pivot from global
candidate search to a local search bounded around the incumbent, on the grounds that three of
four queries had clustered in the same corner without beating a best point supported by a
single observation.

**Rounds 6–8.** Round 5's pivot produced the first record (−0.0325). Round 6 widened the box
and raised β back to 0.8; round 7 switched to a feature-aware search with `compound_3` tightly
bounded to its empirically good band and `compound_1`/`compound_2` left unconstrained, since
their length-scales had pinned at the upper bound; round 8 reverted to a tight ±0.15 local box
and produced −0.0125, a near-threefold improvement.

**Rounds 9–12.** Steady exploitation with the box tightening ±0.15 → ±0.08 → ±0.05 and β
falling 0.3 → 0.2 → 0.15. Round 9 gave −0.003963, round 11 gave **−0.003116**. Round 12 took
a deliberate risk, stepping to `compound_1 = 0.367` to bracket the peak from below; it scored
−0.0202, about 1.6× the fitted noise below prediction. It cost a query and bought the most
useful information of the late rounds: the improvement stops before `compound_1` gets that
low, so the peak lies between round 11's 0.41 and round 8's 0.54.

**Round 13.** The GP's `compound_1`/`compound_2` length-scales remained pinned at their upper
bound of 10, leaving the posterior almost flat in exactly the directions that mattered — about
−0.0101 to −0.0103 at three nearby points that actually scored −0.0040, −0.0101 and −0.0031.
The final query therefore stopped trusting the GP for those two dimensions and fitted a ridge
quadratic along the learned diagonal (PC1, 57% of sampled input variance), with the GP
retained only for `compound_3` and β set to 0. It returned −0.016509 and did not improve.

**What this function taught.** Exploitation is not free of noise either. The best observed
value sits against a posterior mean of about −0.0102 at neighbouring points and a fitted noise
standard deviation of 0.008, so part of it is a favourable draw — which is precisely why the
final query used a smooth fit over five points rather than repeating the best observation.

---

## Function 4 — Warehouse allocation surrogate (4D)

**Result:** −4.025542 → **0.590944** (round 13), a sign change and +4.617 absolute, reaching
81% of the reference best.

**Rounds 1–4.** UCB over the full space. Round 3 returned an extreme value of 2602.25 against
a prior range of −32.6 to −0.066. *(The attribution of this value is unresolved — see
[`datasheet.md`](datasheet.md). The strategy that followed was built on it.)* Round 4 queried
a point 0.041 away and came back ordinary, disproving a broad-basin hypothesis immediately.

**Rounds 5–8.** Chasing the outlier. Round 5 added a compositional kernel (broad RBF plus
narrow RBF) able to represent a smooth background and a sharp local spike simultaneously.
Round 6 committed to a trust-region search around the round 3 point with a novelty constraint
so the query could not merely re-confirm a known location. Round 8 introduced a
**shell-constrained probe** — no farther than 0.008 from the spike, no closer than 0.003 to
any observed point — guaranteeing a genuinely new test closer than anything in rounds 4 to 7.
The trend flattened rather than climbing: 0.561 at distance 0.0116, 0.541 at 0.010, 0.577 at
0.003.

**Rounds 9–11.** Round 9 abandoned hyper-local refinement for a diversification search at
β = 2.5, explicitly excluding everything within 0.15 of the spike. It returned −20.58 and
cost a query. Round 10 reverted to local exploitation centred on the **best confirmed point**
rather than the unconfirmed spike — a distinction made explicit for the first time. Round 11
used k-means on all 40 points to confirm a single hot region, then tested a new angular
direction at the edge of the characterised shell.

**Round 12.** The global GP was by now dominated by the single extreme value, with a
leave-one-out RMSE around 417 for every kernel — a statistical tie that made its suggestions
unstable across machines. It was replaced for this round by a local ridge regression on the
seven observations within 0.06 of the best confirmed point, validated at RMSE 0.08 against
0.57 for a flat-plateau baseline, with a bounded uphill step chosen by bootstrap-ensemble UCB.
The step failed: predicted about 1.32, returned 0.534.

**Round 13.** The failure was diagnosed as curvature — `y` peaks in `x2` near the round 8
value, so moving `x2` down costs value while moving `x4` down still helps. The linear model
was replaced by a local ARD GP fitted only to the eight plateau points, achieving a
leave-one-out RMSE of **0.033** against 0.54 for a flat plateau and 0.36 for the best ridge
fit, and predicting the held-out rounds 4, 11 and 12 to within 0.02–0.03. The local-GP step
was kept only as a benchmark; the query itself was an exploration probe for a second maximum,
0.05–0.12 from the best confirmed point and at least 0.05 from every queried point, with the
rule written down before the result. Forecast 0.364 ± 0.386, observed **0.591** — a new
confirmed best.

**What this function taught.** One extreme reward must be quarantined, or it corrupts every
value estimate downstream. The decision to exclude it from the plateau model was right.
Roughly five rounds spent chasing it were not, and the round-3 attribution issue suggests they
may have been chasing a value belonging to another function entirely.

---

## Function 5 — Chemical process yield (4D)

**Result:** 1088.86 → **8662.405001** (round 13), about 8.0×, reaching 99.999% of the
reference best of 8662.4825.

**Rounds 1–3.** Round 1 was a deliberately exploratory corner query at β = 2.5; the GP had
predicted about 3464 in back-transformed log-space and the true yield was 154 — an early sign
of overconfident extrapolation into an unseen corner. Round 2 found 1909.16. Round 3 pushed
`x2` further in the same direction that had worked and crashed from 1909 to below zero,
revealing a sharp cliff rather than a smooth slope — and exposing a pipeline bug, since
`log1p` is undefined below −1. A signed-log transform replaced it. *(The value that triggered
this is one of the two affected by the round-3 attribution issue; see
[`datasheet.md`](datasheet.md).)*

**Rounds 4–9.** A sequence of narrowing refinements that progressively established the corner
as boundary-pinned. Round 4 showed that small moves in `x1` and `x3` cost about 370 in yield
even with `x2` almost unchanged, which killed the "bracket `x2` only" approach. Round 5 moved
to a tight box across all four dimensions. Round 6 spent a robustness check on a genuinely
different region and returned 68.5. Rounds 8 and 9 established that `x4` behaves exactly like
`x1` and `x3` — a 0.01 move cost about 89 yield — leaving only `x2` with meaningful room.

**Round 10.** The most instructive routine decision of the project. Leave-one-out coverage had
drifted to 82.8%, below the 95% target and alongside round 8's 75.0%, so the surrogate was
overconfident across the dataset rather than at one point. Instead of narrowing further as the
schedule implied, β was *raised* 0.3 → 0.5 and the bounds loosened on all four dimensions. The
result was **1915.964**, a new best — the true `x2` optimum sat a touch further out than the
round 9 box allowed.

**Rounds 11–12.** Round 11 confirmed the corner had plateaued; corner queries span 1912.3 to
1916.0, so `x2` was resolved to about ±0.0005 and further tuning had no measurable value.
Round 12 spent the last free option on a full-domain probe, which found a second region at
about 1085–1089 — real, but lower than the corner, and below the threshold set in advance for
refining it.

**Round 13.** Every query so far had sat in the corner basin or the interior; none at the far
vertex where `x2`, `x3` and `x4` are simultaneously maximal. The GP assigned most unexplored
vertices the same maximum uncertainty (σ ≈ 2.05 in log-space), so the model could not rank
them — this was a hypothesis test, not an optimisation. It returned **8662.405**, about 4.5×
the previous best, against a GP prediction of roughly 124. For reference, the corner ramp
(`y = 2010·x2 + 710` at `x1 = 0`) extrapolates to only about 2720 at `x2 = 1`, so `x1 = 0.999999`
matters a great deal.

**What this function taught.** The acquisition optimiser structurally cannot propose what it
cannot reach. Gradient-based maximisation from interior-seeded starts under-proposes domain
vertices, and nine rounds of refinement inside a committed basin could never have found this.
Vertex candidates have to be added to the candidate set by hand.

---

## Function 6 — Cake recipe scored on five ingredients (5D)

**Result:** −0.714265 → **−0.213554** (round 13), a penalty roughly 70% smaller.

**Rounds 1–4.** Global UCB. Repeated corner-seeking: the acquisition optimiser kept landing
on the same box boundary.

**Rounds 5–7.** Round 5 pivoted to a trust region of radius 0.15 at β = 0.75 and *also*
underperformed, landing on the exact same corner of the trust region as round 4 — so the
boundary-seeking pathology persisted even locally. Round 6 took a manual override, raising
sugar and milk while keeping flour, eggs and butter near the incumbent profile, on a hypothesis
the automated search had no evidence for. It returned **−0.465434**, an improvement of 0.245.
**The hypothesis the repeated automated search kept failing to find was correct, and it was
unfindable from the data the search had gathered.**

**Rounds 8–9.** Round 8 shrank the trust region to 0.07 and replaced UCB with pure mean
exploitation over a dense interior grid, specifically to stop the optimiser landing on edges.
Round 9 kept the edge-avoiding approach but widened the radius to 0.12.

**Rounds 10–11.** Manual radius choice was replaced by an **Expected-Improvement sweep** across
radii from 0.07 to 0.30, taking the radius where EI peaked rather than guessing. EI peaked at
0.20 and was flat out to 0.30, indicating a genuine local mode rather than an artefact of box
size. Round 10 returned **−0.401802**, a new best. Round 11 re-ran the same sweep centred on
the new point; EI now peaked at a much tighter 0.05, confirming refinement within a genuine
local optimum.

**Round 12.** The standard EI sweep still pointed back toward the risky near-edge butter region
at 0.996–0.998 that had already underperformed — the GP's length-scales were long enough that
one bad edge result barely dented its optimism there. Rather than follow EI blindly, a
**capped-EI** variant explicitly bounded butter below the level that had just failed. It
returned **−0.362411**, a new best.

**Round 13.** A data-integrity fix first: the round 11 entry was a copy of round 1's point and
output rather than the real round 11 result, which meant round 12's submitted query had been
computed on a dataset containing a wrong row. The round 12 section was frozen to the data as of
that decision and re-run on corrected data for transparency. Refitting on the corrected 32
points still pushed butter back to about 0.997, while the local evidence pointed the other way:
butter at 0.9996, 0.981 and 0.939 had given −0.520, −0.402 and −0.362. The surrogate was
replaced by a standardised Matérn 5/2 ARD GP (leave-one-out RMSE 0.227 against 0.324; mean
forward error 0.128 against 0.202) whose stated uncertainty was too narrow (forward z up to
−4.0), so it was used inside a 30-member bootstrap ensemble with bag-averaged Expected
Improvement. Result: **−0.213554**, beating both round 12 and the ensemble's own predicted
mean of −0.342.

**What this function taught.** Exploiting a model's confident extrapolation was the real risk,
not exploring. And when a single fit is demonstrably better *and* demonstrably overconfident,
hedging across an ensemble beats choosing one.

---

## Function 7 — ML hyperparameter tuning, 6D

**Result:** 1.364968 → **2.240937** (round 13), +64.2%.

**Rounds 1–4.** A conventional annealing schedule, β 2.0 → 1.2 → 0.8. None of these queries
beat the best point already in the initial data, and all three acquisition-driven proposals
landed on domain boundaries. At 31 points in six dimensions the surrogate could not say where
uncertainty was worth paying for.

**Round 5.** The turning point, and it was evidence-driven. A scan's top candidates differed
from the incumbent almost entirely in `x3`, which the GP had declared irrelevant — and the
observed data contradicted the GP. Round 5 constrained `x3` to an empirically supported band
*against* the surrogate's flatness assumption, moved five coordinates at once, and produced the
largest weekly gain to that point.

**Rounds 6–7.** Rounds that look like failures in the best-so-far curve and were not. They
produced the `x4` boundary finding that explained three earlier disappointments. Round 7
stopped widening and started bracketing: a quadratic through three bracket points put the `x3`
peak at 0.348.

**Rounds 8–12 — the ladder.** Single-variable steps with everything else frozen. Round 8
validated the method (`x3` 0.300 → 0.330, everything else fixed, a cleanly attributable
+0.050179). Round 9 confirmed `x4`'s effect by a controlled pair — same `x3` of 0.360, `x4` at
0.000 gave 1.110 against `x4` at 0.088 giving 1.698, **+0.588 attributable to `x4` alone**.
With no flattening in the gain rate, round 10 scaled the stride from 0.03 to 0.12; round 11
held it, bracketing the peak. By round 12 the gain rate had fallen 45-fold, from +0.0428 to
+0.00096 per 0.03, so the strategy changed from stepping outward to converting the bracket: a
quadratic through the last three rungs gave `x3 = 0.5428` and the GP's single-dimension argmax
gave 0.542. The interpolated vertex beat both bracketing points and closed the ladder.

Along the way, freezing five coordinates narrowed the state distribution sharply. The model
became very accurate locally — leave-one-out mean |z| from 1.56 to 0.89, round 12 error
−0.0032 — and no better globally.

**Round 13 — a rule withdrawn.** A "move type" bias rule, which had treated multi-coordinate
proposals as inherently unreliable since round 8, was tested and failed. All three queries it
rested on also carried a boundary or sub-floor coordinate, a cause already established in round
8, so the two explanations were perfectly confounded in a sample of three. And of the two
multi-coordinate queries with no boundary value, one was round 5 — the largest weekly gain to
that point — which the pooled statistic had concealed. Withdrawing the rule reversed the final
decision: the query became the GP optimum with `x1` held in its proven band, moving six
coordinates at once. It returned **2.240937**, a +0.342 gain, the largest single round of the
project, with the model under-predicting by 0.225.

**What this function taught.** Exploitation of a *method* carries the same risk as exploitation
of a region — it keeps paying until the thing it was built on turns out not to be there. The
controlled single-variable method produced every record and is also precisely what prevented a
wider search: four of six dimensions were probed across a fifth or less of their range, so this
is a well-characterised **local** optimum.

---

## Function 8 — ML hyperparameter tuning, 8D

**Result:** 9.598482 → **10.000000** (round 13), matching the reference best exactly, with the
generating function recovered in closed form.

**Rounds 1–4.** β was the entire strategy: 2.0, 1.5, 1.0, 0.7, with a GP (RBF, ARD) and UCB
over a 10,000-point Latin hypercube. Two improvements, then a stall. Round 3's −1.94σ miss
diagnosed the surrogate as interpolating rather than regularising; raising the noise floor from
1e-8 to 1e-4 put five of the next six predictions inside 1σ.

**Round 5.** Expected Improvement became identically zero everywhere — the model believed
nothing in the space could beat the incumbent. Rather than lower β again, it was raised to 3.0
and a query spent probing 0.23 from the basin. It returned 0.065 *below* the incumbent: a loss
in value that confirmed the basin and validated the calibration fix in the same round. `x8` was
frozen as a confirmed nuisance dimension.

**Rounds 6–7.** Exploitation at β = 0.5 confirmed a plateau; improvements were now smaller than
the model's own σ. Round 7 switched the kernel from RBF to Matérn (ν = 2.5) after systematic
tuning.

**Rounds 8–12 — one factor at a time.** Round 8 abandoned multi-coordinate proposals after four
straight failures and stepped on `x3` alone, targeting the point of maximum disagreement between
the two best-fitting models. It ended the drought. Round 9 spent a query on the strategy's
oldest untested assumption — the `x8` freeze — holding seven coordinates fixed and moving `x8`
from 0.999999 to 0.000000. The result missed by **−4.50σ** and refuted a four-round structural
assumption; `x8`'s length-scale fell from its pinned bound to 232.67. Rounds 10 and 11 stepped
on `x4` and `x5` using the corrected surrogate, both new bests, with round 11 missing by 0.00006
(−0.04σ), the most accurate prediction of the project. Round 12's joint `x1`+`x4` move was
forecast to fail at a gain-to-σ ratio of 0.44, and did, by 0.00012.

**Round 13 — the closed form.** The function was identified as **additively separable quadratic
in 17 parameters**, fitted exactly, and validated to residuals ≤ 1e-15 both in-sample on all 52
prior observations and out-of-sample. Its analytic argmax was submitted and returned exactly
**10.000000**, as predicted to ten decimal places.

**What this function taught, and it is uncomfortable.** Seventeen observations identify the
function exactly, and the initial design supplied forty. The answer was over-determined before
the first query. Twelve rounds of diligent kernel, noise-floor and acquisition tuning missed it
because the search ran inside a model family that could not represent the target — a stationary
kernel with a long length-scale represents a weak additive effect as no effect at all. The GP's
posterior was *well calibrated* at the end and still systematically blind. **A well-calibrated
value estimate inside a misspecified model is exactly as dangerous as a badly calibrated one,
and considerably more persuasive.** What was missing was one round of structural model selection
across model families, at near-zero cost.

---

## What transfers

1. **Validate the model out-of-sample before letting it choose the next experiment.** Every
   improvement in the final two rounds came from a model whose out-of-sample error had been
   measured first.
2. **The size of the prediction error tells you what to update.** Small errors adjust a
   parameter; large ones mean the structure is wrong and no parameter will fix it.
3. **Check the output's dynamic range before fitting anything.** Function 1 lost seven rounds
   to a problem a log transform resolves immediately.
4. **Test model families, not only hyperparameters within one.** Function 8's entire shortfall
   reduces to this.
5. **A productive method deserves the same scepticism as a productive region.** Function 7's
   withdrawn rule and Function 6's capped EI are both cases of refusing to follow a loop that
   had been paying.

   **Note:** Document has been refined using AI
7. **Record the reasoning before the result arrives.** It is the only thing that makes the
   forward test meaningful a week later, and the only defence against rationalising afterwards.
