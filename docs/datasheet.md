# Datasheet: Black-Box Optimisation (BBO) Capstone

**Project: ** Imperial College London - Professional Certificate in Machine Learning and Artificial Intelligence
**Version:** 1.0
**Date   :** 10 October 2026
**Author :** Rishin Mitra

---

## Motivation

### What task does this dataset support?

The dataset records every input submitted to eight black-box objective functions and the
output each one returned. It supports a single task: maximising each function under a hard
budget of one evaluation per function per week, with no access to the function's analytic
form, no gradients, and no ability to batch queries.

It serves a second purpose that became clear during the project. Because every query is
paired with the surrogate model's prediction made *before* submission, the dataset also
records how well a Gaussian Process calibrates itself on sparse data. Several findings in
the weekly reflections — a −1.94σ miss in Function 8's Week 3, a −4.50σ refutation in its
Week 9 — come from that pairing rather than from the objective values themselves.

### Who created it, and why?

Created by Rishin Mitra.

### Was it funded or supported by an organisation?

No external funding. The objective functions are supplied by the course platform; the
query history is the participant's own.

---

## Composition

### What do the instances represent?

Each instance is one evaluation: an input vector in the unit hypercube paired with the
scalar output the black-box function returned. There are no images, documents, or data
about people.

### How many instances are there?

Each function has its own dataset, seeded by a provided initial sample and extended by one
query per week:

| Function | Dimensions | Initial points | Queries appended | Total |
|---|---|---|---|---|
| 1 | 2 | 10 | 9 | 19 |
| 2 | 2 | 10 | 9 | 19 |
| 3 | 3 | 15 | 9 | 24 |
| 4 | 4 | 30 | 9 | 39 |
| 5 | 4 | 20 | 9 | 29 |
| 6 | 5 | 20 | 9 | 29 |
| 7 | 6 | 30 | 9 | 39 |
| 8 | 8 | 40 | 9 | 49 |

Combined: 247 evaluations across eight functions and five distinct dimensionalities. Totals
are the array shapes reported by each notebook at the Week 10 state; the Week 10 query
itself is pending in every function and is not yet appended.

### Is it a sample, and how representative is it?

It is emphatically a sample, and an unrepresentative one by design. The initial points are
quasi-random or space-filling and give reasonable coverage; every query added since is
concentrated near whichever point was best at the time. Section **Gaps** below quantifies
this.

### What data does each instance consist of?

Raw input coordinates and raw outputs, with no labels or annotations. Inputs are
constrained to `[0.000000, 0.999999]` in every dimension and formatted to exactly six
decimal places, which is the submission portal's requirement. Outputs are unbounded reals;
their scale varies enormously between functions — Function 1 operates near 7.71e-16 while
Function 5 operates near 1900 and Function 4 recorded a single value of 2602.

Several functions carry semantic meaning for their inputs. Function 3's three dimensions
are compound doses; Function 6's five are recipe ingredients (flour, sugar, eggs, butter,
milk); Function 8's eight are described in its brief as neural network hyperparameters
(learning rate, batch size, layer count, dropout, regularisation strength, activation,
optimiser type, initial weight range). This matters: an input encoding a *categorical*
quantity such as optimiser type cannot be meaningfully modelled as continuous, and
Function 8's Week 8 reflection identifies exactly that pathology in x8.

### Is any information missing?

No instance has missing fields. What is missing is coverage, and it is substantial:

- **Function 1:** only four of nineteen points are positive — the original best, Week 6
  (5.606704e-16), Week 8 (1.166379e-17) and Week 9 (2.034e-16). Three queries sitting
  0.04–0.09 from the best (Weeks 3, 5 and 7) all returned negative. Domain edges are
  unsampled.
- **Function 2:** sampling concentrated at x1 ≈ 0.65–0.75 over the last five weeks; most
  of the domain below x1 = 0.5 has one or two widely spaced points.
- **Function 3:** compound_3 has stayed in 0.37–0.42 for five weeks; the moderate region
  for compounds 1 and 2 (0.45–0.68, where every recent best sits) is tested at a handful
  of points.
- **Function 4:** roughly a third of all queries sit within a 0.04-radius ball in 4D.
- **Function 5:** about a third within a small box around the Week 2 corner; combinations
  with x1 or x2 large and x3/x4 small are never queried.
- **Function 6:** flour has varied only within 0.39–0.54 across the entire project.
- **Function 7:** the last five queries vary x1 by 0.014, x5 by 0.022 and x6 by 0.026;
  the initial sample's standard deviation is roughly ten times larger in x1 and twenty
  times larger in x5. x1 has never been varied in isolation despite ranking among the most
  influential dimensions.
- **Function 8:** the notebook's own blind-spot diagnostic quantifies this per dimension,
  comparing the spread of all 49 points against the spread of the top ten:

  | dim | ARD length-scale | std (all) | std (top 10) | ratio | risk |
  |---|---|---|---|---|---|
  | x1 | 3.67 | 0.324 | 0.043 | 0.13 | HIGH |
  | x2 | 5.18 | 0.306 | 0.046 | 0.15 | HIGH |
  | x3 | 2.93 | 0.289 | 0.050 | 0.17 | HIGH |
  | x4 | 5.39 | 0.257 | 0.067 | 0.26 | HIGH |
  | x5 | 7.82 | 0.302 | 0.167 | 0.55 | — |
  | x6 | 5.33 | 0.249 | 0.112 | 0.45 | — |
  | x7 | 3.79 | 0.280 | 0.112 | 0.40 | — |
  | x8 | 232.67 | 0.319 | 0.399 | 1.25 | — |

  x1 through x4 now show the same low-variation signature x8 carried before Week 9, which
  the notebook flags directly: their ARD values are the least trustworthy numbers in it.

### Are there relationships between instances?

Yes, and they are not incidental. Every query from Week 2 onward was chosen using a model
fitted to all preceding points, so the instances are sequentially dependent rather than
independently drawn. Function 8's Week 10 reflection names this as a feedback loop: each
query is placed by an acquisition function optimising against the current model, so the
data increasingly confirms what the model already believed.

### Recommended splits

None. The dataset is far too small for train/test splitting; leave-one-out
cross-validation was used throughout instead, and k-fold where leave-one-out became
expensive.

### Does it contain sensitive or identifiable information?

No. All data is synthetic function output. No personal, demographic, or confidential
information is present, and no individual can be identified directly or indirectly.

---

## Collection process

### How was the data acquired?

Initial points were supplied by the course platform as `.npy` files
(`initial_inputs.npy`, `initial_outputs.npy`). All subsequent points were acquired by
submitting a single input vector per function per week through the course portal and
recording the returned value.

### What was the sampling strategy?

It changed deliberately over the ten rounds, and the change is itself part of the record.

**Weeks 1–4 — broad, acquisition-driven.** Queries were produced by maximising an
acquisition function (UCB, later Expected Improvement in some functions) over the full
domain. Exploration weights started high: β = 2.5 in Functions 2, 4, 5 and 6; β = 2.0 in
Functions 7 and 8; Function 1 used an EI exploration floor of 3.0.

**Weeks 5–8 — local and structured.** Exploration weights were reduced as evidence
accumulated, and search was confined to trust regions around the incumbent. Function 1's
exploration floor fell 3.0 → 2.0 → 1.5 → 0.25 → 0.05 → 0.02; Function 5's β ran 2.5 → 0.7 → 2.2
(a deliberate robustness check) → 0.4; Function 8's ran 2.0 → 1.5 → 1.0 → 0.7 → 3.0 → 0.5.
The non-monotonic schedules are not errors — each reversal responded to a specific result.

**Weeks 8–10 — one-factor-at-a-time.** Functions 7 and 8 abandoned multi-coordinate
proposals entirely after observing that every multi-coordinate query produced
uninterpretable results. Both now move a single coordinate per round so that any
discontinuity has an unambiguous cause.

Several queries were chosen by overriding the acquisition function. Function 7's Week 5
constrained x3 manually against the model's belief that x3 was irrelevant, and produced
the project's best result for that function. Function 1 overrode its acquisition function
twice, documenting the reasoning each time.

### Over what time frame?

Ten weeks, one evaluation per function per week. The schedule is the binding constraint on
everything else in this project: it rules out batch evaluation, repeated sampling at a
single point, and any systematic hyperparameter search that would itself require many
trials.

### Was there ethical review or consent?

Not applicable. No human or animal subjects are involved; the objective functions are
synthetic.

---

## Preprocessing, cleaning and labelling

### What preprocessing was done?

Minimal, and the raw data is preserved alongside any transformation.

- **Clipping.** Proposed inputs are clipped to `[0.0, 0.999999]` via `np.clip` before
  submission, since the acquisition optimiser occasionally returns boundary values.
  Function 8 recorded exactly this in Week 2, where the optimiser returned 1.000000 on two
  dimensions — an invalid submission without clipping.
- **Formatting.** All inputs are formatted to six decimal places. Function 6 enforced this
  with explicit `assert` statements before every submission.
- **Signed-log transform.** Function 5 applied one in Week 4. Week 3 returned −1.359780,
  and any value below −1 breaks `log1p`, which the pipeline used; the notebook records this
  as a pipeline bug exposed by the result rather than a modelling choice. This is the only
  transformation applied to outputs.
- **Detrending.** Function 8 removes its measured x8 effect arithmetically after Week 9
  established it is exactly additive. Raw values are retained.

No instances were removed, cleaned, or relabelled.

### Is the raw data preserved?

Yes. Cumulative input and output arrays are saved each week, and every query appears in the
notebook's Progress Log with its raw returned value.

---

## Uses

### What is the dataset intended for?

Reconstructing and auditing the optimisation trajectory for each function; refitting the
surrogates to verify the reported results; and studying how a Gaussian Process behaves when
data is sparse and sequentially dependent.

### What should it *not* be used for?

**Characterising the objective functions themselves.** This is the most important
restriction, and it follows directly from the sampling bias documented above. The data
supports claims about the neighbourhood of each incumbent and almost nothing else. The
course brief notes that one function has two optima, and no query since Week 4 in Function
7 has been more than 0.3 from the current best — nothing in the last six rounds could have
detected a second peak.

**Training a general surrogate.** The selection effect is severe. Function 7's initial
sample is dominated by near-zero outputs while its queries concentrate above 1.1, so any
model fitted to the combined data sees far more detail about the good region than the rest.
That is efficient for optimising and misleading for anything else.

**Inferring feature importance naively.** Function 6 flags a concrete trap: milk and sugar
both correlate negatively with the output across all 29 points, but this is confounded by
the Weeks 2–5 corner queries, which paired low sugar and milk with extreme butter and eggs.
Taking the raw correlation at face value would have argued against a query that the ARD
kernel correctly supported.

### What risks or biases are present?

Three, all documented above: spatial concentration around incumbents, the acquisition-driven
feedback loop, and confounding between dimensions that have never been varied
independently. None involve fairness or harm to people, since no personal data is involved.

---

## Distribution

### How is the dataset available?

Through the project's public GitHub repository, alongside the notebooks that produced it.
Each function's directory contains the original `.npy` seed files, the cumulative arrays
saved weekly, and the notebook whose Progress Log records every query, its output, and the
reasoning behind it.

### Licence and terms of use

The query history and notebooks are released for academic review and reuse. The initial
seed data and the objective functions themselves are the property of the course platform;
redistribution of those is subject to the platform's terms, not the participant's.

### When is it available?

From the point of repository publication, submitted as part of the capstone deliverables.

---

## Maintenance

### Who maintains it?

The capstone participant, through the repository.

### How will it be updated?

Two further rounds remain (Weeks 11 and 12). Each will append one point per function and
update the Progress Log. After the final submission, the dataset is frozen.

### Version control and reproducibility

All randomness is seeded. The seeds differ by function and are set in the notebooks as
follows:

| Function | Seeds in use |
|---|---|
| 1 | `random_state=0` |
| 2 | `random_state=42` |
| 3 | `np.random.seed(42)`, `random_state=0` |
| 4 | `random_state=0`, `seed=42` |
| 5 | `random_state=42`, `default_rng(42)` |
| 6 | `random_state=42`, `default_rng(42)` |
| 7 | `random_state=0`, `random_state=42`, `default_rng(7)` |
| 8 | `random_state=42`, plus `seed` 3, 5, 7 and 11 for stability checks |

Re-running a notebook top to bottom regenerates every number it reports.

**One known reproducibility caveat.** Functions 7 and 8 both observed that proposals differ
across machines in the fifth and sixth decimal places. This traces to multi-start L-BFGS-B
landing in different basins under different BLAS builds, not to a different optimum.
Function 8 measured the discrepancy at up to 8e-5, roughly 250 times smaller than its model
σ of 0.021 — immaterial to results, but it would confuse an exact-match check. Library
versions should be pinned by anyone attempting bitwise reproduction.

### What a reviewer would need

The `.npy` seed files, `plotting_utils.py`, the weekly notebooks, and the written
reflections. The last of these is not optional: several decisions — the delta and beta
schedules in Function 3 (0.15 → 0.08 → 0.05 and 0.3 → 0.2 → 0.15), Function 1's
`xi_multiplier` schedule (3.0 → 2.0 → 1.5 → 0.25 → 0.05 → 0.02), Function 8's ±0.08
one-factor window and 0.35 blind-spot threshold, Function 7's choice of x3 = 0.48 over 0.39
or 0.52 — reflect judgement about risk and remaining budget rather than any formula. The
code shows *what* was queried; only the prose explains why that direction was chosen over
the alternatives.
