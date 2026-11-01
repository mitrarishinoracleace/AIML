# Methodology

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 1.0 |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |

---

This document describes *how* each query was produced: the per-round pipeline, how the
surrogate was fitted and selected, how acquisition functions were optimised, and what had to
pass before a point was submitted. For *what* was decided round by round see
[`decision_log.md`](decision_log.md); for the per-function narrative see
[`strategy.md`](strategy.md).

---

## 1. The problem as posed

Eight objective functions, 2 to 8 dimensions, each a sealed system. No analytic form, no
gradients, no batching. Inputs bounded to `[0.000000, 0.999999]` per dimension and submitted
as a hyphen-separated string of six-decimal values. One query per function per week, for
thirteen weeks. Each function is seeded with a supplied sample of 10 to 40 points.

The grading uses the best input/output pair found overall, not the value of the final query.
This is a load-bearing detail. It makes every query a free option in the sense that a poor
result cannot lose ground already banked, which is why several late queries are deliberate
probes rather than refinements — and why the *final* query is the one place where pure
exploitation is strictly correct, since no future query exists to benefit from what it
teaches.

---

## 2. Per-round pipeline

The same seven steps ran every round, in this order. Steps 2 to 5 are the model; steps 1, 6
and 7 are what makes the result auditable.

### Step 1 — Append and verify the previous result

The previous round's input and output are appended to the dataset, and the arrays are
re-checked for shape and alignment. This step is not clerical. It caught a corrupted
Function 3 output in round 5 and a duplicated Function 6 row in round 13, and it is where
the unresolved round-3 attribution between Functions 4 and 5 became visible.

### Step 2 — Score the previous prediction before fitting anything

Before refitting, the prediction made for the previous query — computed without that point —
is compared against what was actually returned, as a signed error and as a z-score against
the model's own stated σ. This is the **forward test**, and it is the single most useful
diagnostic in the project, because it is out-of-sample in time as well as in space. Most
strategy changes in the decision log were triggered by this number rather than by the
objective value.

### Step 3 — Refit and re-select the kernel

A kernel bake-off is re-run on the full dataset every round rather than fixed once:

- **Candidates:** `ConstantKernel × Matérn(ν ∈ {1.5, 2.5}, ARD) + WhiteKernel` against
  `ConstantKernel × RBF(ARD) + WhiteKernel`, with per-function length-scale and noise bounds.
- **Criterion:** leave-one-out cross-validated RMSE, preferred over log-marginal likelihood.
  The two disagree, and the disagreement matters: Function 8's RBF had the higher in-sample
  likelihood while Matérn generalised better, and round 8 vindicated the choice when RBF
  predicted the winning point would be *worse* than the incumbent.
- **Tie-break:** from Function 4's round 12 onward, when leave-one-out RMSEs fall within 1%
  the kernel is chosen by a fixed priority order (Matérn, RBF, composite) rather than by
  numerical noise, so the selection is stable across machines.
- **Hyperparameters:** length-scales, amplitude and noise are fitted internally by
  gradient-based marginal-likelihood maximisation with `n_restarts_optimizer=10`. With up to
  eight ARD length-scales the likelihood surface is multi-modal and a single start lands in
  poor optima.

### Step 4 — Validate the surrogate before trusting it

Three checks, run every round:

| Check | What it measures | Target |
|---|---|---|
| Leave-one-out RMSE | Accuracy on held-out points | Comparable to fitted noise |
| Leave-one-out mean &#124;z&#124; | Calibration | ≈ 0.8 |
| Interval coverage | Honesty of the stated uncertainty | 95% |

A model that fails these is not used to choose a query. Function 5 raised β and loosened its
search bounds in round 10 *because* coverage had fallen to 82.8%, rather than narrowing
further as its schedule implied — and the next result was a new best. Function 6's final
round abandoned a single fit for a 30-member bootstrap ensemble because the better-fitting
model had forward z-scores reaching −4.0.

### Step 5 — Optimise the acquisition function over a bounded region

Two acquisition functions were used, with a third option in the final round:

- **Upper Confidence Bound:** maximise `μ(x) + β·σ(x)`.
- **Expected Improvement:** maximise `E[max(0, y − y*)]` under the posterior, with an
  exploration floor applied through a scaled `ξ`.
- **Ensemble / bagged EI:** average EI across bootstrap refits, used where a single fit was
  demonstrably overconfident.

Maximisation uses multi-start L-BFGS-B from a fixed Latin-hypercube design, typically 20
restarts, constrained to a per-function search region: the full domain early, a trust region
around the incumbent later, and in some functions a feature-aware box with some coordinates
pinned.

**Known limitation of this step.** A gradient-based maximiser over an interior-seeded design
structurally under-proposes domain vertices. Function 5's final result — 8662.4 at
`(0.999999)⁴`, against a GP prediction of about 124 — was reachable only by proposing that
vertex by hand. Where the optimum may sit on a boundary, vertex candidates must be added to
the candidate set explicitly.

### Step 6 — Apply submission constraints

- Clip to `[0.0, 0.999999]` with `np.clip`. The acquisition optimiser does return exact
  boundary values; Function 8 recorded 1.000000 on two dimensions in round 2, which would
  have been an invalid submission.
- Format to exactly six decimal places and join with hyphens. Function 6 enforces this with
  explicit `assert` statements before every submission.
- Check novelty where required — several functions imposed a minimum distance from every
  previously queried point, so a query could not merely re-confirm a known location.

### Step 7 — Record the decision before the result exists

The progress log entry is written now: the query, the model's prediction and σ, the
exploration weight, the search region, the kernel chosen, and the reason this point was
preferred to the alternatives. Writing it before the outcome is known is what makes the
forward test in step 2 meaningful a week later, and it is the only defence against
rationalising a decision after seeing how it turned out.

---

## 3. Exploration policy

### What was tried first

A scheduled annealing of the exploration weight — β wound down round by round on a calendar.
Function 2 ran 2.5 → 0.15, Function 8 ran 2.0 → 1.5 → 1.0 → 0.7, Function 1 ran an EI
exploration floor from 3.0 down to 0.02.

### Why it was abandoned

The schedule assumes the surrogate can say where uncertainty is worth paying for. At 10 to
40 points in 2 to 8 dimensions it frequently could not. All three of Function 7's
acquisition-driven proposals landed on domain boundaries; Function 3's five corner queries
scored between −0.050 and −0.082 and its near-origin query returned its worst value of
−0.173; Function 6's trust-region search repeatedly landed on the exact same corner of the
box, a boundary-seeking pathology that persisted even locally.

### What replaced it

Exploration became **conditional rather than scheduled** — explore when the model is
unreliable, exploit when a mechanism has been demonstrated. Four mechanisms implemented this:

**Trust regions sized by evidence.** Narrow after a good result, widen after a poor or
uncertain one. Function 6 eventually replaced manual radius choice entirely with an
Expected-Improvement sweep across radii from 0.07 to 0.30, taking the radius where EI peaked;
that change found its first new best in four rounds.

**One factor at a time.** From round 8, Functions 7 and 8 moved a single coordinate with the
rest frozen. This is maximally exploitative in the frozen dimensions and genuinely
exploratory in the moved one, and it makes every gain individually attributable — Function 7
measured +0.588 from `x4` alone by a controlled pair, and +0.050 then +0.051 per 0.03 step
along `x3`. Its cost is that it cannot detect interactions, and it produced the narrow
coverage recorded in [`limitations.md`](limitations.md).

**An explicit risk threshold.** Function 8 formalised the decision as a **gain-to-σ ratio**:
a direction earns a query only when its predicted gain clears the model's own uncertainty.
Across six tests the rule discriminated cleanly — 1.45, 0.87 and 0.81 produced new bests;
0.44, 0.29 and 0.11 failed. Round 12 was forecast to fail at 0.44 and did.

**Free-option reasoning near the end.** Because grading uses the best pair found, a late
query that fails costs nothing relative to the banked incumbent. Functions 2, 5 and 6 each
spent their second-to-last query on a deliberate probe for this reason. The final query
inverts it: with no future query to inform, the value of information is zero, so Functions 1,
3 and 4 used pure exploitation and Function 1 formalised it as *maximise Expected Improvement
subject to P(beat best) ≥ 50%* rather than taking the unconstrained EI maximum, which carried
only about a 22% chance.

---

## 4. When to change the model rather than the query

The project's most consequential methodological rule emerged late and is stated here
explicitly, because it is what produced six of the eight final records.

**The size of the prediction error determines what to update.** A small error means a
parameter needs adjusting. A large one means the model structure is wrong, and adjusting
parameters inside it will not help.

Worked examples:

| Function | Error | Response |
|---|---|---|
| 2, round 12 | −0.6σ | None. Beliefs left intact |
| 8, round 3 | −1.94σ | Raise the noise floor 1e-8 → 1e-4. Five of the next six predictions landed inside 1σ |
| 2, round 11 | −5.7σ | Refit; fitted noise rose sevenfold |
| 8, round 9 | −4.50σ | Overturn a four-round structural assumption about `x8` |
| 1, round 11 | 3.8e10× overshoot | Replace the model. The GP was fitted on a scale where the target was invisible |
| 4, round 12 | predicted 1.32, got 0.534 | Replace a straight-line model with one carrying curvature |

The diagnostic that should have existed from round 1, and did not, is a **standing
model-class comparison**: fit an additive model, a low-order polynomial and a full GP, and
compare leave-one-out error every round. It costs almost nothing. Had it run on Function 8,
it would have flagged the additive structure immediately — a leave-one-out RMSE at machine
precision is not a tuning success, it means the generating function has been recovered.

---

## 5. Validation protocol

Four layers, in increasing order of how much they are trusted:

1. **In-sample fit** — reported, never used to justify a decision. Function 8's noise-floor
   fix deliberately made it worse.
2. **Leave-one-out cross-validation** — RMSE, mean |z| and coverage. Used for kernel
   selection and as a calibration check.
3. **Forward test** — refit on everything before round *k*, predict round *k*. The most
   informative measure, and the one that drove most strategy changes.
4. **Controlled single-variable experiment** — hold every coordinate fixed, move one, attribute
   the difference. The only measurement here that supports a causal claim, and the basis for
   Function 7's `x4` finding and Function 8's `x8` refutation.

A fifth check applies where a closed form is available: Function 8's recovered model was
validated against all 52 prior observations *and* out-of-sample, both to residuals ≤ 1e-15,
before its analytic argmax was submitted.

### Statistical caution applied to the results themselves

Two guards were used against over-reading a favourable draw:

- **Compare gains against fitted noise.** Function 3's late improvements of 0.003 to 0.007
  are smaller than its fitted noise standard deviation of 0.008, so its best observed value
  is partly a lucky draw — which is why the final query used a smooth fit over five ridge
  points rather than repeating the best observation.
- **Require enough episodes to separate competing explanations.** Function 7's withdrawn
  move-type rule rested on three failed queries in which two candidate causes were perfectly
  confounded. Three is not enough, and a pooled statistic can conceal the single episode that
  would discriminate.

---

## 6. Reproducibility

- **Seeds are fixed** per notebook: `random_state=42` predominantly, with `random_state=0`,
  `default_rng(7)` and additional seeds 3, 5, 7 and 11 used for stability checks.
- **A single reusable `bayesian_optimization_step()`** per notebook consolidates GP fitting
  and acquisition maximisation, so the same code path produced every query.
- **Historical rounds are anchored to the data available at the time.** Function 1's rounds 1
  to 12 run on a 21-point snapshot, its round 13 decision on 22 points, and its closing
  section on all 23. Without this, appending later observations would silently change what
  earlier cells report, and the queries actually submitted would no longer reproduce.
- **Two known caveats.** Proposals differ across machines in the fifth and sixth decimal
  places, traced to multi-start L-BFGS-B landing in different basins under different BLAS
  builds — Function 8 measured the discrepancy at up to 8e-5, roughly 250 times smaller than
  its model σ of 0.021. And `requirements.txt` is unpinned and omits `seaborn`, which all
  eight notebooks import.

---

## 7. What this methodology would change next time

Stated as changes, not regrets, and each traceable to something that went wrong:

1. **Run a model-class comparison every round**, not kernel comparison within one class. This
   is the single highest-value change and it would have solved Function 8 in round 1.
2. **Transform the output before fitting, not after a failure.** Function 1 lost seven rounds
   to a scale problem that a log or signed-log transform resolves immediately. Check the
   dynamic range of the output first.
3. **Add domain vertices to every candidate set.** Function 5's result sat at one, and no
   gradient-based maximiser would ever have proposed it.
4. **Budget exploration structurally rather than by schedule.** Reserving a fixed fraction of
   queries for maximum-uncertainty sampling is robust to the failure seen here, where a
   productive exploit loop crowded exploration out entirely.
5. **Plan over the remaining horizon.** Standard acquisition functions are myopic. Rounds 11
   to 13 were planned by hand as a sequence — informative query, convert the bracket, then
   exploit — which is what a non-myopic acquisition function does automatically.
6. **Condition diagnostics on enough episodes to be identifiable.** Function 7's withdrawn
   rule is the warning: a confident correction derived from a confounded sample of three is
   worse than no correction at all.

---

See [`strategy.md`](strategy.md) for how this was applied per function,
[`decision_log.md`](decision_log.md) for the dated record of changes,
[`model_cards.md`](model_cards.md) for model details and
[`limitations.md`](limitations.md) for the scope of the resulting claims.

**Note:** Document has been refined using AI
