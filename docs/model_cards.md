# Model Card: Black-Box Optimisation (BBO) Capstone

**Project: ** Imperial College London - Professional Certificate in Machine Learning and Artificial Intelligence
**Version:** 1.0
**Date   :** 10 October 2026
**Author :** Rishin Mitra

---

## Model overview

- **Name:** GP-BO Surrogate Optimiser
- **Type:** Gaussian Process regression surrogate with acquisition-driven sequential search
- **Task:** maximise eight unknown black-box functions of 2 to 8 dimensions under a budget
  of one evaluation per function per round
- **Kernels:** Matérn (ν = 1.5 and 2.5) and RBF, both with Automatic Relevance
  Determination, plus a WhiteKernel noise term. Selection varies by function and was made
  on evidence: Functions 1, 3, 4, 7 and 8 settled on Matérn (Function 3 by leave-one-out
  RMSE, 0.0803 against RBF's 0.0836); Functions 2, 5 and 6 settled on RBF, Function 5
  re-confirming it on the full 29-point dataset
- **Acquisition:** Upper Confidence Bound in most functions, with Expected Improvement
  adopted in Functions 1, 6, 7 and 8
- **Implementation:** scikit-learn `GaussianProcessRegressor`; SciPy `minimize` (L-BFGS-B)
  with multi-start for acquisition optimisation; `scipy.stats.norm` for EI;
  `scipy.stats.qmc.LatinHypercube` for candidate generation

---

## Intended use

### Suitable for

Expensive-evaluation optimisation in low dimensions where each query costs real time or
money, calibrated uncertainty is required by the acquisition step, and the available
sample is in the tens rather than the thousands. This is the regime Bayesian optimisation
was designed for and where it has decades of use in engineering design.

### Target users

Researchers and practitioners running physical experiments, clinical or materials trials,
or hyperparameter searches where a full training run is the unit of cost.

### Use cases to avoid

**Global optimisation claims.** The approach finds and refines a basin; it does not
establish that the basin is global. Seven of eight functions have searched locally since
roughly Week 4, and Function 7 states the consequence plainly: if its function is the
two-optima case the brief mentions, this approach will not find the second peak, and no
amount of refinement inside the current basin will reveal it.

**Discontinuous or strongly non-stationary surfaces.** A stationary kernel cannot
represent an effect that is sharp locally and absent on average. This failed in practice
more than once — see **Failure modes** below.

**Categorical inputs encoded as continuous.** Function 8's x8 pinned to 0.999999 in Week 2
and 0.000000 in Week 3 while its length-scale saturated at its upper bound. If x8 encodes
optimiser type, as the function's brief suggests, that is precisely the pathology expected
when a continuous surrogate is imposed on a categorical input: there is no interior optimum
to find, so the search slides to whichever endpoint is marginally favoured. Tree-structured
Parzen estimators or random-forest surrogates handle mixed spaces natively; a GP does not.

**High dimensions with a large budget.** Exact GP inference is O(n³). The approach has
perhaps a few hundred more points of headroom before sparse or inducing-point
approximations become necessary.

---

## Approach details

### How decisions are made

Each round fits a GP to all accumulated observations, selects a kernel by cross-validated
error, optimises an acquisition function over a bounded candidate region, and submits the
resulting point. Every acquisition computation is printed before being acted on rather than
asserted, and the reasoning is recorded before the result arrives.

### Evolution across ten rounds

**Rounds 1–4: broad acquisition-driven search.** High exploration weights over the full
domain. This phase performed poorly in several functions — Function 7's first four rounds
produced nothing that beat its initial data, and Function 6 repeatedly re-derived the same
failed corner.

**Rounds 5–7: local trust regions and tuned exploration.** Search was confined to boxes
around the incumbent, with exploration weights reduced as confidence grew. Schedules were
non-monotonic by design: Function 5 ran β 2.5 → 0.7 → 2.2 → 0.4, raising it deliberately in
Week 6 as a robustness check; Function 8 ran 2.0 → 1.5 → 1.0 → 0.7 → 3.0 → 0.5, raising it
in Week 5 after exploitation stalled at a cost of 0.065 in forgone value. The rule applied
throughout was to narrow after a good result and widen after a poor or uncertain one.

**Rounds 8–10: one-factor-at-a-time.** The decisive methodological change. Functions 7 and
8 both observed that multi-coordinate proposals produced their worst results and taught
nothing when they failed, because no single cause could be isolated. Both switched to
moving one coordinate per round. Function 8's Week 8 ended a four-week drought this way and
produced a clean attribution at the same time; Function 7's last three records all came from
single-variable steps.

### Model selection and tuning

Kernel choice was made by evidence rather than assumption. Function 7 compared Matérn
against RBF by five-fold cross-validation every week and Matérn won every time. Function 8
selected on leave-one-out error rather than log-marginal-likelihood, deliberately: RBF had
the higher in-sample likelihood but Matérn generalised better, and Week 8 vindicated the
choice when RBF predicted the winning point would be *worse* than the incumbent.

Continuous hyperparameters (length-scales, amplitude, noise) were fitted internally by
gradient-based marginal-likelihood maximisation with `n_restarts_optimizer=10`, since with
eight ARD length-scales the likelihood surface is multi-modal and a single start lands in
poor optima. Discrete choices — kernel family, acquisition type — were compared by grid,
since no gradient exists between "RBF" and "Matérn".

---

## Performance

### Metrics used

- **Best-so-far value** — the headline objective
- **Leave-one-out RMSE** — surrogate accuracy on held-out points
- **Leave-one-out mean |z|** — calibration; approximately 0.8 indicates a well-calibrated model
- **Interval coverage** — proportion of predictions falling inside their stated interval,
  against a 95% target
- **Predicted-versus-actual, logged every round** — so optimism is visible rather than inferred

### Results

| Fn | Dim | Pts | Initial best | Current best | Gain |
|---|---|---|---|---|---|
| 1 | 2 | 19 | 7.71e-16 | 7.71e-16 | none |
| 2 | 2 | 19 | 0.611205 | 0.711209 | +0.100 |
| 3 | 3 | 24 | −0.0348 | −0.003963 | +0.0308 |
| 4 | 4 | 39 | ~ −0.07 | 0.576869 | see note |
| 5 | 4 | 29 | 1088.860 | 1915.052 | +76% |
| 6 | 5 | 29 | −0.714265 | −0.465434 | +0.249 |
| 7 | 6 | 39 | 1.364968 | 1.698012 | +24% |
| 8 | 8 | 49 | 9.598482 | 9.976026 | +0.378 |

Trajectories differ in shape. Function 1 ran nine queries without beating its provided data
and holds four positive points in total. Function 2 took its largest single-round gain
(+0.091) in Week 9, immediately after a Week 8 miss. Function 3 sat flat for seven weeks
before jumping in Week 8 (−0.0125) and again in Week 9. Function 5 gained 75% in Week 2
(1909.163) and then ran a long tail, with Week 7 setting the record. Function 7 made two
discrete jumps and then climbed a ladder: 1.365 → 1.597 → 1.647 → 1.698. Function 8's gains
fell steadily: +0.2897 → +0.0512 → +0.0344 → +0.0023.

Three results deserve flagging rather than burying in the table. **Function 1 has not
improved on its provided data in nine rounds** — every reported refinement describes the
peak's shape, not its height. **Function 4's headline number is a 2602.253793 outlier from
Week 3 that has never been reproduced**, against a prior range of −32.6 to −0.066; the
confirmed plateau at 0.576869 is what the notebook treats as the reportable result.
**Function 6's best was set in Week 6 and Weeks 7, 8 and 9 all fell below it** (−0.628456,
−0.640971, −0.551631), so its recent trajectory is recovery rather than progress.

### Calibration

This improved where it was explicitly targeted, and degraded where it was not. Function 8
raised its noise floor from 1e-8 to 1e-4 in Week 5, deliberately making the model fit its
training data *worse* by stopping it interpolating. Function 7's leave-one-out mean |z|
improved 1.561 → 0.986 once its fitted noise fell from 0.0999 to 1e-06 and all six
length-scales became finite again.

Function 5 moves the other way and is the honest counter-example: its full-dataset
leave-one-out coverage **dropped to 82.8%** by Week 10, below the 95% target and alongside
Week 8's 75.0%. The notebook reads this as the surrogate being overconfident across the
dataset rather than at one point, and responds by raising β from 0.3 to 0.5 and loosening
bounds on all four dimensions instead of narrowing further.

### Diminishing returns

Quantified most precisely in Function 8: successful queries returned +0.2897, +0.0512,
+0.0344 and +0.0023, roughly a 129-fold reduction, with the dataset growing 40 → 48 points
for a 0.02% gain in objective value. By Week 9 the best available refinement gain (+0.00036)
had fallen below the model's own σ at that point (0.00147) — the signal was smaller than
the noise floor of the instrument measuring it.

---

## Assumptions

Five load-bearing assumptions, each of which shaped results:

**Determinism** (Functions 1 and 4). Repeated queries at the same point would return the
same value. This was never tested — every query has been at a distinct point, since
spending a round on reproducibility was never affordable. If the functions have a
stochastic component, Function 1's monotonic-decay reasoning may partly reflect noise, and
Function 4's Week 3 spike of 2602 could be a single noisy draw at an ordinary point rather
than a real feature.

**Smoothness** (Functions 2 and 6). The GP assumes the function varies gradually within the
fitted length-scale. Function 6 demonstrated the failure mode directly in Week 8, predicting
−0.380680 and receiving −0.640971.

**Unimodality** (Function 5). Stated in the brief and leaned on heavily since Week 4, when
three dimensions were pinned and only the fourth refined. If false, the approach converges
confidently to a local optimum.

**Separability** (Function 7). That moving x3 along a line through the current best traces
the same shape it would trace elsewhere. The GP does not share this assumption — it fits
interactions — and there is direct evidence against it: the x4 effect is enormous near zero
and modest elsewhere, which is itself an interaction with the domain boundary.

**Stationarity** (Function 8). The sharpest of the five, and the one that cost most. A
stationary kernel with a long length-scale represents a linear additive effect as no effect
at all. As the Week 10 reflection puts it: the model was not fitting a weak trend badly, it
was modelling it as absent.

---

## Limitations and failure modes

### The documented failure

Function 8 froze x8 for four rounds on the belief it was irrelevant. Its ARD length-scale
sat pinned at 1000, the model was calibrated, cross-validation was healthy, and every
diagnostic agreed. Week 9 tested it directly: holding the other seven coordinates fixed and
moving x8 from 0.999999 to 0.000000 took the output from 9.976026 to 9.956026, against a
prediction of 9.975402 ± 0.00431 — a miss of −0.019376, or **−4.50σ**. Refitting with that
observation dropped x8's length-scale from its pinned 1000 to 232.67.

Two lessons follow. First, a long ARD length-scale does not mean a dimension is
unimportant — it means the likelihood found no evidence of importance, which conflates "this
does not matter" with "you never showed me the contrast." x8 was pinned purely because every
high-scoring point happened to sit at the same x8 value. Second, every diagnostic was wrong
together, because all were downstream of the same data and the same structural assumption.
The freeze was right in outcome, since x8's optimum was at the boundary anyway — but for the
wrong reason, and had x8 trended downward the same logic would have locked the search at the
worst end of that axis.

### Structural limitations

**Budget against dimensionality.** 49 points in eight dimensions, 39 in six and in four,
19 in two. The results are locally credible and globally unsupported; they should be
reported as the best point found in one well-characterised basin, not as the function's
maximum.

**The approach does not always beat the provided data.** Function 1 is the clearest case:
nine rounds of increasingly fine local refinement, with the exploration floor cut from 3.0
to 0.02, produced four positive points and no improvement on the initial best. Three
queries placed 0.04–0.09 from that best all returned negative. Where the peak is narrower
than the search can resolve, the method characterises it without climbing it.

**The one-query-per-week cadence** makes it structurally impossible to separate "is this an
anomaly" from "is this a real narrow feature," since that requires repeated queries at the
same point. Function 4 carries this forward as an open question rather than a resolved
result.

**The feedback loop.** Every query since Week 4 was placed by an acquisition function
optimising against the current model, so the data increasingly confirms what the model
already believed. Function 8's only two deliberate breaks were Week 5's distant probe and
Week 9's boundary test — and Week 9 is the one that overturned an assumption. That is not a
coincidence worth ignoring.

**Self-limiting methodology.** The one-factor-at-a-time method that produced every recent
gain buys interpretability by moving one coordinate at a time, which is exactly why it
cannot detect interactions between coordinates.

---

## Ethical considerations

No personal data is involved and no decisions about people are made, so the usual fairness
concerns do not apply. Two considerations do.

**Transparency as a precondition for reproducibility.** Every query, output, exploration
weight and strategy change sits in a Progress Log; the reflections record why each choice
was made *before* the result arrived, so reasoning cannot be rationalised after the fact;
seeds are fixed throughout. A reviewer needs the `.npy` files, `plotting_utils.py` and the
notebooks to regenerate every number — plus the written reflections, because several
thresholds were set by judgement after inspecting a sweep and are shown but never derived.
Naming that gap is part of the documentation rather than an admission that undermines it.

**Honest reporting of negative results.** Weeks 6 and 7 look like failures in Function 7's
best-so-far curve, but they produced the x4 finding that explained three earlier
disappointments. A reviewer reading only the trajectory would miss why the strategy changed.
Reporting the reversals — freezing x8 and then unfreezing it, raising β when the schedule
said lower it — is what makes the final result interpretable rather than merely favourable.

### Is this card sufficient?

For its scope, yes. It states what the approach does, where it fails, and which assumptions
carry the weight — and the failure it documents most fully is one of its own. What it
deliberately does not claim is a global optimum for any function, which is the single
statement a reader is most likely to want and the one the evidence cannot support.

Two additions would improve it if the project continued: per-function calibration tables
rather than the summary given here, and a formal reproducibility test across machines to
bound the BLAS-related variation rather than merely note it.
