# Model Card: Black-Box Optimisation (BBO) Capstone

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 2.0  |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |

---

> Follows the structure proposed in Mitchell et al. (2019), *Model Cards for Model
> Reporting*, FAccT 2019, 220–229.

---

## Model overview

- **Name:** GP-BO Surrogate Optimiser
- **Type:** Gaussian Process regression surrogate with acquisition-driven sequential search,
  supplemented in the final rounds by parametric models fitted to the same data
- **Task:** maximise eight unknown black-box functions of 2 to 8 dimensions under a budget
  of one evaluation per function per round, over thirteen rounds
- **Kernels:** Matérn (ν = 1.5 and 2.5) and RBF, both with Automatic Relevance
  Determination, plus a `WhiteKernel` noise term and a `ConstantKernel` scale factor.
  Selection was re-run every round on cross-validated error rather than fixed once
- **Acquisition:** Upper Confidence Bound and Expected Improvement, varying by function and
  by round; replaced entirely in three functions by the final round
- **Implementation:** scikit-learn `GaussianProcessRegressor`; SciPy `minimize` (L-BFGS-B)
  with multi-start for acquisition optimisation; `scipy.stats.norm` for EI;
  `scipy.stats.qmc.LatinHypercube` for candidate generation

### Per-function summary at close

| Fn | Dim | Points | Model at close | Acquisition at close | Best observed | Round |
|---|---|---|---|---|---|---|
| 1 | 2 | 23 | Two-source log-magnitude parametric model | Monte-Carlo EI, constrained to P(beat) ≥ 50% | 4.240382e-05 | 13 |
| 2 | 2 | 23 | RBF ARD GP | Space-filling tie-break (models indifferent) | 0.711209 | 9 |
| 3 | 3 | 28 | Matérn 3/2 ARD GP + ridge quadratic along PC1 | Quadratic peak, β = 0 | −0.003116 | 11 |
| 4 | 4 | 43 | Local Matérn ARD GP on plateau points only | Mean + 2σ exploration probe | 0.590944 | 13 |
| 5 | 4 | 33 | Matérn/RBF ARD GP under signed-log transform | Hard vertex probe | 8662.405001 | 13 |
| 6 | 5 | 33 | 30-member bootstrap ensemble of standardised Matérn 5/2 ARD GPs | Bag-averaged EI | −0.213554 | 13 |
| 7 | 6 | 43 | Matérn 5/2 ARD GP | GP optimum with `x1` held in its proven band | 2.240937 | 13 |
| 8 | 8 | 53 | Additively separable quadratic, 17 parameters, solved exactly | Analytic argmax | 10.000000 | 13 |

Three of the eight finished on something other than a Gaussian Process. That is the single
most important fact in this card, and the section **Where the GP was replaced** explains why.

---

## Intended use

### Suitable for

Expensive-evaluation optimisation in low dimensions where each query costs real time or
money, calibrated uncertainty is required by the acquisition step, and the available sample
is in the tens rather than the thousands. This is the regime Bayesian optimisation was
designed for and where it has decades of use in engineering design.

### Target users

Researchers and practitioners running physical experiments, clinical or materials trials,
or hyperparameter searches where a full training run is the unit of cost.

### Use cases to avoid

**Global optimisation claims.** The approach finds and refines a basin; it does not
establish that the basin is global. Seven of the eight functions searched locally from
roughly round 4 onward, and the results should be read as the best point found in one
well-characterised region. The exception is Function 8, where the generating function was
recovered in closed form and its optimum is therefore verifiable rather than merely
observed.

**Discontinuous or strongly non-stationary surfaces.** A stationary kernel cannot represent
an effect that is sharp locally and absent on average, nor one whose scale varies by many
orders of magnitude across the domain. Both failures occurred — see **Failure modes**.

**Objectives whose output scale spans many orders of magnitude, without transformation.**
Function 1 is the cautionary case. Fitted on raw `y`, the GP's variance was set by a single
−3.6e-03 reading, roughly twelve orders of magnitude larger than the target basin's own
spread of about 3e-16. The basin was not modelled badly; it was numerically invisible.

**Categorical inputs encoded as continuous.** Function 8's `x8` toggled between 0.999999
and 0.000000 while its ARD length-scale saturated. If `x8` encodes optimiser type, as the
brief suggests, that is the pathology expected when a continuous surrogate is imposed on a
categorical input. Tree-structured Parzen estimators or random-forest surrogates handle
mixed spaces natively; a GP does not.

**High dimensions with a large budget.** Exact GP inference is O(n³). The approach has
perhaps a few hundred more points of headroom before sparse or inducing-point
approximations become necessary.

---

## Approach details

### How decisions are made

Each round fits a GP to all accumulated observations, re-selects the kernel by
cross-validated error, optimises an acquisition function over a bounded candidate region,
and submits the resulting point. Every acquisition computation is printed before being
acted on rather than asserted, and the reasoning is recorded in the progress log *before*
the result arrives, so it cannot be rationalised afterwards.

### Evolution across thirteen rounds

**Rounds 1–4 — broad acquisition-driven search.** High exploration weights over the full
domain. This phase performed poorly almost everywhere: Function 7's first four rounds
produced nothing that beat its initial data and all three acquisition-driven proposals
landed on domain boundaries; Function 3's five corner queries scored between −0.050 and
−0.082; Function 6 repeatedly re-derived the same failed corner.

**Rounds 5–7 — local trust regions and tuned exploration.** Search was confined to boxes
around the incumbent, with exploration weights reduced as confidence grew. Schedules were
non-monotonic by design: Function 5 ran β 2.5 → 0.7 → 2.2 → 0.4, raising it deliberately as
a robustness check; Function 8 ran 2.0 → 1.5 → 1.0 → 0.7 → 3.0 → 0.5, raising it after
Expected Improvement collapsed to identically zero across the whole domain. The rule applied
throughout was to narrow after a good result and widen after a poor or uncertain one.

**Rounds 8–12 — one factor at a time.** Functions 7 and 8 moved a single coordinate per
round, with the rest frozen, so that any change in output had an unambiguous cause. Function
8 ended a four-round drought this way and produced a clean attribution at the same time;
Function 7 climbed a measured ladder, 1.365 → 1.597 → 1.647 → 1.698 → 1.869 → 1.873 → 1.899.
Function 6 meanwhile replaced manual radius choice with an Expected-Improvement sweep over
trust-region radii, which found its first new best in four rounds.

**Round 13 — structural change.** Five functions replaced the model class rather than
retuning it. This phase produced six of the eight final records.

### Model selection and tuning

Kernel choice was made by evidence rather than assumption, and re-made every round. Function
3 selected Matérn over RBF on leave-one-out RMSE in most rounds, by margins as small as
0.0749 against 0.0818 on the final 27-point dataset. Function 8 selected on leave-one-out
error rather than log-marginal likelihood, deliberately: RBF had the higher in-sample
likelihood but Matérn generalised better, and round 8 vindicated the choice when RBF
predicted the winning point would be *worse* than the incumbent. Function 4 added an
explicit tie-break rule in round 12 — when leave-one-out RMSEs fall within 1%, choose by a
fixed priority rather than by numerical noise.

Continuous hyperparameters (length-scales, amplitude, noise) were fitted internally by
gradient-based marginal-likelihood maximisation with `n_restarts_optimizer=10`, since with
eight ARD length-scales the likelihood surface is multi-modal and a single start lands in
poor optima. Discrete choices — kernel family, acquisition type — were compared by grid,
since no gradient exists between "RBF" and "Matérn".

---

## Where the GP was replaced

**Function 1 — scale, not search.** Eleven rounds passed without improving on the initial
best. The diagnosis arrived in round 11, when the closest query yet to the incumbent
returned the second-*lowest* basin output while the GP had predicted a 3.8e10× overshoot.
The GP was replaced by a parametric log-magnitude model: first a local log-space Gaussian
fitted to the six basin points, which reproduced all six to 2% and ranked them exactly by
distance to a fitted centre, then a global two-source model that placed a stronger source
beyond the sampled region. The round 13 query was chosen by maximising Expected Improvement
under that model subject to at least a 50% chance of beating the incumbent, with the unknown
sign-flip radius integrated out by Monte Carlo. It predicted 2.56e-05 against an actual
4.24e-05 — a ratio of 0.60, the best out-of-sample prediction any model made in this project.

**Function 8 — model family, not hyperparameters.** After twelve rounds of kernel,
noise-floor and acquisition tuning, the function was identified as additively separable
quadratic in 17 parameters and solved in closed form. The recovered model reproduces all 52
prior observations and holds out-of-sample to residuals ≤ 1e-15; its analytic argmax
returned exactly 10.000000. Seventeen observations suffice to identify it and forty were
supplied before the first query, so the answer was over-determined from the start. What was
missing was never more data or better planning — it was one round of *structural* model
selection across model families rather than hyperparameter tuning within one.

**Function 6 — one fit replaced by an ensemble.** The standardised Matérn 5/2 fit had lower
leave-one-out error (RMSE 0.227 against 0.324 for the previous RBF GP) and lower mean
forward error (0.128 against 0.202), but its stated uncertainty was too narrow — forward z
reached −4.0. Rather than trust a single better-fitting but overconfident model, the final
query averaged Expected Improvement over a 30-member bootstrap ensemble plus the full fit,
hedging between the surrogate's mode and the previously validated region.

**Functions 3, 4 and 5 — supplemented rather than replaced.** Function 3 kept its GP for
one dimension and fitted a ridge quadratic along a learned diagonal for the other two, after
the GP's length-scales pinned at their upper bound left it almost flat exactly where
resolution was needed. Function 4 fitted a local ARD GP to the eight plateau points only,
with the outlier excluded, achieving a leave-one-out RMSE of 0.033 against 0.54 for a
flat-plateau baseline and 0.36 for the best ridge fit. Function 5 kept its GP but overrode
the acquisition optimiser, which structurally favours interior points, to probe a domain
vertex directly.

---

## Performance

### Metrics used

- **Best-so-far value** — the headline objective
- **Leave-one-out RMSE** — surrogate accuracy on held-out points
- **Leave-one-out mean |z|** — calibration; approximately 0.8 indicates a well-calibrated model
- **Interval coverage** — proportion of predictions falling inside their stated interval,
  against a 95% target
- **Forward (out-of-sample) error** — refit on everything before round *k*, predict round
  *k*, compare. Genuinely out-of-sample in time as well as space, and the more informative
  of the two error measures
- **Predicted-versus-actual, logged every round** — so optimism is visible rather than inferred

### Results

| Fn | Dim | Points | Initial best | Final best | Gain | Reference best | Versus reference |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 23 | 7.710875e-16 | **4.240382e-05** | ~5.5e10× | 3.50e-06 | ~12× higher |
| 2 | 2 | 23 | 0.611205 | **0.711209** | +16.4% | 0.668242 | +6.4% |
| 3 | 3 | 28 | −0.034835 | **−0.003116** | ~11× smaller penalty | −7.172e-03 | ~2.3× better |
| 4 | 4 | 43 | −4.025542 | **0.590944** | +4.617 | 0.730826 | 81% of it |
| 5 | 4 | 33 | 1088.86 | **8662.405001** | ~8.0× | 8662.4825 | 99.999% of it |
| 6 | 5 | 33 | −0.714265 | **−0.213554** | penalty ~70% smaller | −0.068174 | penalty ~3.1× larger |
| 7 | 6 | 43 | 1.364968 | **2.240937** | +64.2% | 3.267413 | 68.6% of it |
| 8 | 8 | 53 | 9.598482 | **10.000000** | +4.18% | 10.000000 | matched exactly |

All eight improved on their initial best. Six set their record in the final round.

### Trajectory shapes

They differ enough that a single convergence story would be misleading.

- **Function 1** — flat for eleven rounds, then two jumps totalling about 5.5e10×, both
  produced by changing the model rather than the search region.
- **Function 2** — a single step in round 9 (+0.091, immediately after a round 8 miss) and
  nothing afterwards. Four subsequent queries, including the final one, failed to beat it.
- **Function 3** — flat for seven rounds, then −0.0125, −0.0040 and −0.0031 in rounds 8, 9
  and 11. The last two differ by less than the fitted noise of 0.008.
- **Function 5** — a 75% gain in round 2, a long plateau at about 1915, then a 4.5× jump at
  a vertex no acquisition optimiser had proposed.
- **Function 7** — the most consistent: two discrete jumps, then a measured ladder, then the
  largest single-round gain of the project (+0.342) in round 13.
- **Function 8** — strictly diminishing gains (+0.2897, +0.0512, +0.0344, +0.0045, +0.0024)
  followed by a discontinuity when the closed form was found.

### Calibration

Calibration improved where it was explicitly targeted and degraded where it was not.

Function 8 raised its noise floor from 1e-8 to 1e-4 in round 5, deliberately fitting its
training data *worse* by stopping the model interpolating; five of the next six predictions
landed inside 1σ, and its round 11 prediction missed by 0.00006 (−0.04σ), the most accurate
of the project. Function 7's leave-one-out mean |z| improved from 1.561 to 0.986 once its
fitted noise fell from 0.0999 to 1e-06 and all six length-scales became finite again.

Two honest counter-examples. Function 5's full-dataset leave-one-out coverage dropped to
82.8% by round 10, below the 95% target and alongside round 8's 75.0%; the notebook read
this as overconfidence across the dataset rather than at one point and responded by *raising*
β from 0.3 to 0.5 and loosening bounds instead of narrowing further — a decision the round 10
result then validated with a new best. Function 6's leave-one-out |z| exceeded 2 for roughly
one point in five and its forward z reached −4.0, which is why its final query used an
ensemble rather than a single fit.

The most important calibration finding is Function 8's, and it is uncomfortable. By round
11 the GP was *well* calibrated — predictions landing inside their stated intervals, errors
of the right magnitude — and still systematically blind, because a stationary kernel with a
long length-scale represents a weak additive effect as no effect at all. **A well-calibrated
value estimate inside a misspecified model is exactly as dangerous as a badly calibrated
one, and considerably more persuasive.**

### Decision rules that were measured, not assumed

Function 8 formalised its risk decision as a **gain-to-σ ratio**: a direction earns a query
only when its predicted gain clears the model's own uncertainty. The rule discriminated
cleanly across six tests — ratios of 1.45, 0.87 and 0.81 all produced new bests, while 0.44,
0.29 and 0.11 all failed. Round 12 was forecast as likely to fail at 0.44 and did, missing
by 0.00012.

### Diminishing returns

Quantified most precisely in Function 8: successful queries returned +0.2897, +0.0512,
+0.0344 and +0.0023, roughly a 129-fold reduction, with the dataset growing from 40 to 48
points for a 0.02% gain in objective value. By round 9 the best available refinement gain
(+0.00036) had fallen below the model's own σ at that point (0.00147) — the signal was
smaller than the noise floor of the instrument measuring it.

That is the correct reading of a refinement loop, and it is also the trap. The remaining
0.017 to the true optimum was not reachable by any amount of further refinement, and was
reached in one step by changing the model.

---

## Assumptions

Five load-bearing assumptions, each of which shaped results. Three were tested by the end.

**Determinism.** Repeated queries at the same point would return the same value. No point
was ever queried twice, so this was never tested directly — spending a round on
reproducibility was never affordable. Indirect evidence is now strong: Function 5's output
is exactly linear in `x2` along the corner ridge (R² = 0.999995) across queries taken weeks
apart, and Function 8's recovered closed form reproduces every observation to 1e-15. Apparent
noise in the other functions is better read as model misfit than measurement error.

**Smoothness.** The GP assumes the function varies gradually within the fitted length-scale.
Function 6 demonstrated the failure directly in round 8, predicting −0.380680 and receiving
−0.640971.

**Unimodality** (Function 5). Stated in the brief and leaned on heavily from round 4, when
three dimensions were pinned and only the fourth refined. The final round found 8662.4 at a
vertex while the long-refined corner plateaued at 1915.9 — so either the brief's unimodality
refers to a structure the corner search never reached, or the assumption held while the
*region* was wrong. Either way the lesson is the same: pinning three of four dimensions from
round 4 made the remaining nine rounds unable to discover this.

**Separability** (Function 7). That moving one coordinate along a line through the current
best traces the same shape it would trace elsewhere. The GP does not share this assumption —
it fits interactions — and there is direct evidence against it: the `x4` effect is enormous
near zero and modest elsewhere, which is itself an interaction with the domain boundary.

**Stationarity** (Function 8). The sharpest of the five, and the one that cost most. It was
not merely imperfect; it was false in a way no amount of data inside that model class could
reveal. The function is additively separable, which a stationary ARD kernel can approximate
but never identify.

---

## Limitations and failure modes

### Three documented failures

**Function 8 — a diagnostic that was right about the symptom and wrong about the cause.**
`x8` was frozen for four rounds on the belief it was irrelevant. Its ARD length-scale sat
pinned at its upper bound, the model was calibrated, cross-validation was healthy, and every
diagnostic agreed. Round 9 tested it directly: holding the other seven coordinates fixed and
moving `x8` from 0.999999 to 0.000000 took the output from 9.976026 to 9.956026, against a
prediction of 9.975402 ± 0.00431 — a miss of −0.019376, or **−4.50σ**. Refitting dropped
`x8`'s length-scale to 232.67, and the round 13 closed form confirmed a real, weak, additive
effect.

Two lessons follow. A long ARD length-scale does not mean a dimension is unimportant; it
means the likelihood found no evidence of importance, which conflates "this does not matter"
with "you never showed me the contrast." And every diagnostic was wrong together, because
all were downstream of the same data and the same structural assumption.

**Function 1 — exploiting a model that could not see the target.** Seven rounds of
increasingly fine local refinement, with the exploration floor cut from 3.0 to 0.02, moved
the best value not at all. What looked like exploitation was repeated sampling of a slope.
The failure was invisible from inside the model: the GP's predictions were wrong by three to
ten orders of magnitude and nothing in its own diagnostics said so, because the residuals it
was minimising lived on a scale where the basin did not exist.

**Function 7 — a rule derived from confounded evidence.** From round 8 a bias rule treated
multi-coordinate proposals as inherently unreliable, on the evidence of three failed queries.
The round 13 audit found all three also carried a boundary or sub-floor coordinate — a cause
already established in round 8 — so the two explanations were perfectly confounded in a
sample of three. Worse, the pooled statistic concealed the one episode that discriminated
between them: round 5 moved five coordinates at once, kept every one off the floor, and
produced the largest weekly gain to that point. The rule was withdrawn, which reversed the
final decision and produced a +0.342 gain. **Misattributing a reward signal to a spurious
feature is a standard failure, and the correction required returning to the raw episodes
rather than trusting the summary.**

### Structural limitations

**Budget against dimensionality.** 53 points in eight dimensions, 43 in six and in four, 23
in two. Results are locally credible and globally unsupported, except where a closed form
makes a stronger claim available.

**Coverage is uneven by construction.** The methods that produced the improvements —
trust regions, frozen coordinates, local refinement — are precisely what prevented a wider
search. Four of Function 7's six dimensions and four of Function 8's eight were moved across
a fifth or less of their range.

**The feedback loop.** Every query from round 4 onward was placed by an acquisition function
optimising against the current model, so the data increasingly confirms what the model
already believed. Function 8's round 11 clustering found exactly one dense region, composed
entirely of its own queries, with all forty initial points classified as noise. Its only two
deliberate breaks from the loop were round 5's distant probe and round 9's boundary test —
and round 9 is the one that overturned an assumption.

**One query per round** makes it structurally impossible to separate "is this an anomaly"
from "is this a real narrow feature," since that requires repeated queries at the same point.

**Self-limiting methodology.** One-factor-at-a-time buys interpretability by moving one
coordinate at a time, which is exactly why it cannot detect interactions between coordinates.

---

## Ethical considerations

No personal data is involved and no decisions about people are made, so the usual fairness
concerns do not apply. Three considerations do.

**Transparency as a precondition for reproducibility.** Every query, output, exploration
weight and strategy change sits in a progress log; the reasoning for each choice was recorded
*before* the result arrived; seeds are fixed throughout. A reviewer needs the `.npy` files,
`plotting_utils.py` and the notebooks to regenerate every number — plus the written
reflections, because several thresholds were set by judgement after inspecting a sweep and
are shown but never derived. Naming that gap is part of the documentation rather than an
admission that undermines it.

**Honest reporting of negative results.** Rounds 6 and 7 look like failures in Function 7's
best-so-far curve, but they produced the `x4` finding that explained three earlier
disappointments. Function 8's round 5 probe cost 0.065 in forgone value and confirmed the
basin. A reviewer reading only the trajectory would miss why the strategy changed. Reporting
the reversals — freezing `x8` and then unfreezing it, raising β when the schedule said lower
it, deriving a rule and then withdrawing it — is what makes the final result interpretable
rather than merely favourable.

**Reporting a data defect that reflects badly on the work.** One round-3 output has an
unresolved attribution between Functions 4 and 5, documented in full in the
[datasheet](datasheet.md). It changes no final result but it means two intermediate
narratives rest on a value that may not belong to the function that used it. It is recorded
rather than quietly corrected, because a reader cannot assess the trajectory without it.

### Is this card sufficient?

For its scope, yes. It states what the approach does, where it fails, and which assumptions
carry the weight — and the failures it documents most fully are its own. What it deliberately
does not claim is a global optimum for any function except Function 8, where the recovered
closed form supports the claim.

Two additions would improve it if the project continued: per-function calibration tables
rather than the summary given here, and a formal reproducibility test across machines to
bound the BLAS-related variation rather than merely note it.

---

## References

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E.,
  Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *FAccT 2019*, 220–229.
  https://doi.org/10.1145/3287560.3287596
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.
  MIT Press.
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient Global Optimization of
  Expensive Black-Box Functions. *Journal of Global Optimization*, 13(4), 455–492.
- Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian Process Optimization
  in the Bandit Setting. *ICML 2010*, 1015–1022.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of
  Machine Learning Algorithms. *NeurIPS 25*, 2951–2959.

See the [project README](../README.md) for results in context, [`datasheet.md`](datasheet.md)
for the data, [`methodology.md`](methodology.md) for the pipeline, and
[`limitations.md`](limitations.md) for the full statement of scope.
