# Datasheet: Black-Box Optimisation (BBO) Capstone

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 2.0 (final submission) |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |


> Follows the structure proposed in Gebru et al. (2021), *Datasheets for Datasets*,
> *Communications of the ACM* 64(12), 86–92.

---

## Motivation

### What task does this dataset support?

The dataset records every input submitted to eight black-box objective functions and the output each one returned. It supports a single task: maximising each function under a hard
budget of one evaluation per function per week, with no access to the function's analytic form, no gradients, and no ability to batch queries.

It serves a second purpose that became clear during the project. Because every query is paired with the surrogate model's prediction made *before* submission, the dataset also
records how well a Gaussian Process calibrates itself on sparse data. Several of the most consequential findings come from that pairing rather than from the objective values
themselves — a −1.94σ miss in Function 8's round 3 that diagnosed the surrogate as interpolating rather than regularising, a −4.50σ result in its round 9 that overturned a
four-week-old structural assumption, and a 3.8e10× over-prediction in Function 1's round 11 that revealed the surrogate was fitted on a scale where the target signal was invisible.

In three cases the paired record supported something stronger than calibration analysis. Function 8's inputs and outputs turned out to identify the generating function exactly: it
is additively separable quadratic in 17 parameters, recoverable from 17 observations, and 40 were supplied before the first query. Function 1's outputs, re-modelled in log-space,
identify a two-source structure that the raw-scale data conceals entirely.

### Who created it, and why?

Created by Rishin Mitra, as the capstone deliverable for the programme - "Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence".

### Was it funded or supported by an organisation?

No external funding. The objective functions and the initial observations are supplied by the course platform; the query history is the participant's own.

---

## Composition

### What do the instances represent?

Each instance is one evaluation: an input vector in the unit hypercube paired with the scalar output the black-box function returned. There are no images, documents, or data
about people.

### How many instances are there?

Each function has its own dataset, seeded by a provided initial sample and extended by one query per round over thirteen rounds:

| Function | Dimensions | Initial points | Queries appended | Total | Best observed |
|---|---|---|---|---|---|
| 1 | 2 | 10 | 13 | 23 | 4.240382e-05 (round 13) |
| 2 | 2 | 10 | 13 | 23 | 0.711209 (round 9) |
| 3 | 3 | 15 | 13 | 28 | −0.003116 (round 11) |
| 4 | 4 | 30 | 13 | 43 | 0.590944 (round 13) |
| 5 | 4 | 20 | 13 | 33 | 8662.405001 (round 13) |
| 6 | 5 | 20 | 13 | 33 | −0.213554 (round 13) |
| 7 | 6 | 30 | 13 | 43 | 2.240937 (round 13) |
| 8 | 8 | 40 | 13 | 53 | 10.000000 (round 13) |

**Combined: 279 evaluations** — 175 supplied initial points plus 104 queries — across eight functions and five distinct dimensionalities. The dataset is now closed; no further queries
will be appended.

### Is it a sample, and how representative is it?

It is emphatically a sample, and an unrepresentative one by design. The initial points are quasi-random or space-filling and give reasonable coverage of each domain. Almost every
query added afterwards is concentrated near whichever point was best at the time. The **Gaps and coverage** subsection below quantifies this per function, and the effect is
severe enough that it constrains what the dataset can legitimately be used for. 

### What data does each instance consist of?

Raw input coordinates and raw outputs, with no labels or annotations. Inputs are constrained to `[0.000000, 0.999999]` in every dimension and formatted to exactly six 
decimal places, which is the submission portal's requirement. Outputs are unbounded reals, and their scale varies enormously between functions: Function 1 spans roughly 121 orders of
magnitude and finishes near 4.2e-05, while Function 5 finishes at 8662.4 and Function 3 is negative throughout by construction.

Several functions carry semantic meaning for their inputs. Function 3's three dimensions are compound doses; Function 6's five are recipe ingredients (flour, sugar, eggs, butter,
milk); Function 8's eight are described in its brief as neural network hyperparameters (learning rate, batch size, layer count, dropout, regularisation strength, activation,
optimiser type, initial weight range). This matters, because an input encoding a *categorical* quantity such as optimiser type cannot be meaningfully modelled as
continuous. Function 8 flagged exactly that pathology in `x8`, whose ARD length-scale had run to 232 against neighbours of 3 to 5, and read it as irrelevance. A controlled
single-variable test in round 9 contradicted that at −4.50σ, and the round 13 closed-form solve settled it: `x8` does enter the function, additively and weakly. "Irrelevant" was the
wrong reading of a real signal.

### Output sign and range, per function

| Function | Positive outputs | Full observed range |
|---|---|---|
| 1 | 15 of 23 | −3.606e-03 to 4.240e-05 |
| 2 | 19 of 23 | −0.0656 to 0.7112 |
| 3 | 0 of 28 | −0.3989 to −0.003116 |
| 4 | 8 of 43 | −32.63 to 0.5909 |
| 5 | 33 of 33 | 0.1129 to 8662.41 |
| 6 | 0 of 33 | −2.5712 to −0.2136 |
| 7 | 43 of 43 | 0.0027 to 2.2409 |
| 8 | 53 of 53 | 5.5922 to 10.0000 |

Functions 3 and 6 are negative by design — both are penalty scores where the objective is to approach zero from below. Function 1's negative readings are not errors either: the
round 13 modelling treats the sign as physical, with an estimated sign-flip radius of 0.057 to 0.078 around the fitted source.

### Gaps and coverage

No instance has missing fields. What is missing is coverage, and it is substantial. The clearest measure is the span each dimension was given by the thirteen queries, against the
span the initial sample already covered:

| Function | Query span per dimension | Initial span (typical) | Queries within 0.1 of the final best |
|---|---|---|---|
| 1 | 0.409, 0.832 | ~0.80 | 7 of 13 |
| 2 | 0.910, 1.000 | ~0.74–0.90 | 4 of 13 |
| 3 | 0.866, 0.921, 0.850 | ~0.72–0.93 | 5 of 13 |
| 4 | 0.403, **0.088**, 0.584, 0.285 | ~0.90–0.95 | 10 of 13 |
| 5 | 1.000, 0.986, 0.929, 0.688 | ~0.72–0.89 | 1 of 13 |
| 6 | 0.536, 0.458, 0.632, **0.161**, 0.268 | ~0.82–0.96 | 1 of 13 |
| 7 | **0.067**, 0.425, 1.000, **0.199**, **0.058**, **0.096** | ~0.89–0.98 | 1 of 13 |
| 8 | **0.174**, **0.157**, **0.106**, **0.163**, 0.299, 0.228, 0.250, 1.000 | ~0.89–0.98 | 1 of 13 |

Bold entries mark dimensions the queries moved across a fifth or less of their available range. Four of Function 7's six dimensions and four of Function 8's eight fall into that
category — in both cases this is a direct consequence of the controlled single-variable strategy that produced their improvements. The method that generated the results is the
same method that prevented a wider search.

Two refinements of that picture are worth stating:

- **Function 7** — the three dimensions held nearly fixed (`x1`, `x5`, `x6`) were frozen   deliberately from round 8 onward so that any change in output had an unambiguous cause.
  The dimension given the widest span (`x3`, full range) is the one that was walked.
- **Function 8** — the apparent full span in `x8` is misleading. It reflects a binary   toggle between 0 and 0.999999 across several rounds rather than a sweep, and `x8` was
  frozen entirely in others.

The low-variation signature below was Function 8's own round-11 blind-spot diagnostic, comparing the spread of all 49 points then available against the spread of the top ten:

| dim | ARD length-scale | std (all) | std (top 10) | ratio | flagged |
|---|---|---|---|---|---|
| x1 | 3.67 | 0.324 | 0.043 | 0.13 | HIGH |
| x2 | 5.18 | 0.306 | 0.046 | 0.15 | HIGH |
| x3 | 2.93 | 0.289 | 0.050 | 0.17 | HIGH |
| x4 | 5.39 | 0.257 | 0.067 | 0.26 | HIGH |
| x5 | 7.82 | 0.302 | 0.167 | 0.55 | — |
| x6 | 5.33 | 0.249 | 0.112 | 0.45 | — |
| x7 | 3.79 | 0.280 | 0.112 | 0.40 | — |
| x8 | 232.67 | 0.319 | 0.399 | 1.25 | — |

**Resolution.** The diagnostic was right about the symptom and wrong about the remedy. It correctly identified that the ARD values for `x1` to `x4` were the least trustworthy numbers
in the notebook, but the implied fix — sample those dimensions more widely — was unnecessary. The round 13 structural solve recovered the function exactly from data already
in hand, to residuals ≤ 1e-15 both in-sample and out-of-sample. The coverage gap was real; it simply was not the binding constraint.

### Are there relationships between instances?

Yes, and they are not incidental. Every query from round 2 onward was chosen using a model fitted to all preceding points, so the instances are sequentially dependent rather than
independently drawn. Function 8's round 11 clustering analysis makes the consequence concrete: DBSCAN found exactly one dense region in the combined dataset, composed
**entirely** of model-guided queries, with all 40 initial points classified as noise. The data increasingly describes where the model already believed the answer was.

### Recommended splits

None. The dataset is far too small for train/test splitting. Leave-one-out cross-validation was used throughout, supplemented by a forward test — refit on everything available before
round *k*, then compare the prediction for round *k* against what was actually returned. That forward record is the more informative of the two, because it is genuinely
out-of-sample in time as well as in space.

### Does it contain sensitive or identifiable information?

No. All data is synthetic function output. No personal, demographic, or confidential information is present, and no individual can be identified directly or indirectly.

---

## Errors, noise and known issues

### Are the functions noisy?

They appear to be deterministic. The clearest evidence is Function 5, where output is exactly linear in `x2` along the corner ridge (R² = 0.999995) across queries taken weeks
apart. No point was ever queried twice, so this was not verified directly for every function, and several rounds were spent learning the distinction between genuine noise and
model error. Where a fitted `WhiteKernel` reports a non-zero noise level, that figure is better read as the scale of the surrogate's own misfit than as measurement noise.

### One round-3 output has an unresolved attribution

This is the dataset's most significant known defect and it is documented here in full.

In `data/accumulated/week3/outputs.txt`, six of the eight values match the corresponding notebook exactly. The remaining two are shifted by one position against each other:

| Slot in the portal file | Value | Used by |
|---|---|---|
| Function 4 | 0.5572213082937378 | no notebook |
| Function 5 | 2602.2537934860957 | the Function 4 notebook |
| Function 6 | −1.359780271551876 | the Function 5 **and** Function 6 notebooks |

Two functions cannot both own the same observation, so at least one reading is wrong. The portal ordering is the more likely one on three grounds: 2602 lies inside Function 5's
observed range (0.11 to 8662) and three orders of magnitude outside Function 4's (−32.6 to 0.59); 0.557221 sits squarely inside the plateau at 0.534 to 0.591 that Function 4 settled
on; and Function 3 independently identified and corrected this same off-by-one in round 5, recording that it had wrongly used 0.5572213082937378 as its own round-3 output.

**No final best is affected.** Function 4's record (0.590944) and Function 5's (8662.405001) were both set in round 13. The consequences are to the trajectory, not the
result: under the portal reading, Function 4's rounds 4 to 9 were spent probing a value that does not belong to it, and Function 5's best-so-far between rounds 3 and 12 is
understated in its notebook. Two downstream narratives inherit the ambiguity — Function 4's "unconfirmed spike" and the signed-log transform Function 5 introduced in round 4 to handle
a value below −1 — and both are flagged where they appear.

Pending resolution with the course platform, the portal files are treated as authoritative for reported convergence, and each affected notebook retains the value it was built on so
that its historical cells still reproduce.

### Other known issues

- **Function 6** carried a duplicated row at round 11 — a copy of the round 1 observation   had been appended in place of the round 11 result. This was found and corrected during
  the round 13 audit. The corrected value is in the current dataset.
- **Function 3** corrected its round 3 output at source in round 5, as described above. 
- No instances were removed, imputed, or relabelled for any other reason.

---

## Collection process

### How was the data acquired?

Initial points were supplied by the course platform as `.npy` files (`initial_inputs.npy`, `initial_outputs.npy`). All subsequent points were acquired by
submitting a single input vector per function per round through the course portal, as a hyphen-separated string of six-decimal values, and recording the returned scalar.

### What was the sampling strategy?

It changed deliberately across the thirteen rounds, and the change is itself part of the record.

**Rounds 1–4 — broad and acquisition-driven.** Queries were produced by maximising an acquisition function (UCB, and Expected Improvement in some functions) over the full
domain. Exploration weights started high: β = 2.5 in Functions 2, 3, 4, 5 and 6; β = 2.0 in Functions 7 and 8; Function 1 used an EI exploration floor of 3.0. The results argued
against the approach. Function 3's five corner queries scored between −0.050 and −0.082 and its query near the origin returned −0.173, the worst value it ever recorded; Function 7's
proposals landed on domain boundaries and none beat the best point already in its initial data.

**Rounds 5–8 — local and structured.** Exploration weights were reduced as evidence accumulated, and search was confined to trust regions around the incumbent. Function 1's
exploration floor fell 3.0 → 2.0 → 1.5 → 0.25 → 0.05 → 0.02; Function 5's β ran 2.5 → 0.7 → 2.2 (a deliberate robustness check) → 0.4; Function 8's ran 2.0 → 1.5 → 1.0 → 0.7 → 3.0 →
0.5. The non-monotonic schedules are not errors — each reversal responded to a specific result, and Function 8's rise to 3.0 came after Expected Improvement collapsed to
identically zero across the whole domain.

**Rounds 8–12 — one factor at a time.** Functions 7 and 8 moved a single coordinate per round, with the remainder frozen, so that any change in output had an unambiguous cause.
This produced all six of Function 7's records and three of Function 8's five improvements, and it is the direct cause of the narrow coverage documented above.

**Round 13 — structural change.** Five functions replaced their model class rather than retuning it: Function 1 substituted a parametric two-source log-magnitude model for the GP,
Function 8 solved its recovered closed form analytically, Function 3 fitted a ridge quadratic along a learned diagonal, Function 6 averaged Expected Improvement over a 30-model
bootstrap ensemble, and Function 5 probed an unexplored domain vertex that no gradient-based acquisition maximiser would have proposed.

Several queries were chosen by overriding the acquisition function, and these are flagged in each notebook's progress log. Function 7's round 5 constrained `x3` manually against the
model's belief that `x3` was irrelevant, and produced that function's largest early gain.
Function 6's round 6 raised sugar and milk on a hypothesis the automated search had no evidence for, and improved on the incumbent by 0.245.

**One rule was derived and then withdrawn.** From round 8, Functions 7 and 8 treated multi-coordinate proposals as inherently unreliable, on the evidence of three failed
queries. Function 7's round 13 audit found that all three also carried a boundary or
sub-floor coordinate — a cause already established in round 8 — so the two explanations
were perfectly confounded in a sample of three, and the one episode that discriminated
between them (round 5, five coordinates moved at once, the largest weekly gain to that
point) had been hidden by the pooled statistic. The rule was withdrawn, which reversed
the final query decision; the resulting six-coordinate move produced the project's largest
single-round gain. Any analysis that treats "multi-coordinate moves fail" as a property of
this dataset is repeating a mistake its own authors corrected.

### Over what time frame?

Thirteen weeks, one evaluation per function per week. The schedule is the binding constraint
on everything else in this project: it rules out batch evaluation, repeated sampling at a
single point, and any systematic hyperparameter search that would itself require many
trials.

### Was there ethical review or consent?

Not applicable. No human or animal subjects are involved; the objective functions are
synthetic.

---

## Preprocessing, cleaning and labelling

### What preprocessing was done?

Minimal, and the raw data is preserved alongside every transformation.

- **Clipping.** Proposed inputs are clipped to `[0.0, 0.999999]` via `np.clip` before
  submission, since the acquisition optimiser occasionally returns boundary values.
  Function 8 recorded exactly this in round 2, where the optimiser returned 1.000000 on two
  dimensions — an invalid submission without clipping.
- **Formatting.** All inputs are formatted to six decimal places. Function 6 enforced this
  with explicit `assert` statements before every submission.
- **Signed-log transform.** Function 5 applied one from round 4, after a value below −1
  broke the `log1p` call its pipeline used. The notebook records this as a pipeline bug
  exposed by a result rather than a modelling choice. (The value that triggered it is one
  of the two affected by the round-3 attribution issue above.)
- **Log-magnitude modelling.** Function 1 models `ln|y|` from round 12 onward, with sign
  handled separately as a physical property of the source. On the raw scale its basin
  occupies a spread of about 3e-16 inside an overall spread of about 8e-4, so the basin is
  numerically invisible to a model fitted on raw `y`.
- **Standardisation.** Function 6 standardises outputs before fitting, which prevents
  unexplored regions of an all-negative objective from scoring above observed ones.
- **Detrending.** Function 8 removed its measured `x8` effect arithmetically after round 9
  established it was exactly additive. Raw values are retained throughout, and the round 13
  closed-form solve uses them unmodified.

No instances were removed, cleaned, or relabelled, other than the two corrections recorded
under **Errors, noise and known issues**.

### Is the raw data preserved?

Yes. Cumulative input and output arrays are saved for every round under
`data/accumulated/week<1-13>/`, and every query appears in its function's progress log with
the raw returned value, the prediction made beforehand, and the reasoning for the choice.
Each notebook under `notebooks/Final/function<1-8>/` additionally carries its observations
inline, so it runs standalone without reading the `data/` tree.

---

## Uses

### What is the dataset intended for?

Reconstructing and auditing the optimisation trajectory for each function; refitting the
surrogates to verify the reported results; and studying how a Gaussian Process behaves when
data is sparse and sequentially dependent. Function 8 additionally supports a cleaner use
than any of these: because the generating function has been recovered exactly, its 53
observations form a small but exactly-labelled benchmark against which a surrogate's
out-of-sample error can be measured without ambiguity.

### What should it *not* be used for?

**Characterising the objective functions themselves.** This is the most important
restriction, and it follows directly from the sampling bias documented above. For six of the
eight functions the data supports claims about the neighbourhood of each incumbent and
little else. Function 7 is the clearest case: four of its six dimensions were moved across a
fifth or less of their range, every high-performing observation sits in one narrow slice,
and the result should be read as a well-characterised **local** optimum rather than a global
one. The exceptions are Function 8, where the recovered closed form makes global statements
verifiable, and Function 1, whose final two rounds deliberately extrapolated about 5σ
beyond the sampled region.

**Training a general surrogate.** The selection effect is severe. Function 7's initial
sample is dominated by near-zero outputs while its queries concentrate above 1.1, so any
model fitted to the combined data sees far more detail about the good region than about the
rest. That is efficient for optimising and misleading for anything else.

**Inferring feature importance naively.** Three concrete traps are documented in the
notebooks. Function 6 shows that milk and sugar both correlate negatively with output across
all observations, but the correlation is confounded by the early corner queries, which
paired low sugar and milk with extreme butter and eggs. Function 8 shows an ARD length-scale
of 232 on `x8` being read as irrelevance, when the closed-form solve later confirmed a real
if weak additive effect. Function 7 shows a bias rule derived from three failed queries
that survived until the final round and then collapsed under scrutiny. In a dataset this
small and this self-selected, a summary statistic will often describe the sampling policy
rather than the function.

**Treating reported uncertainty as reliable everywhere.** Function 6's leave-one-out |z|
exceeds 2 for roughly one point in five and its forward z reached −4.0; Function 5's
leave-one-out coverage fell to 75% near the corner against a nominal 95%. Where the
posterior is overconfident, any acquisition value computed from it is optimistic.

### What risks or biases are present?

Four, all documented above: spatial concentration around incumbents; the acquisition-driven
feedback loop, which Function 8's clustering analysis quantifies directly; confounding
between dimensions that were never varied independently; and the unresolved round-3
attribution. None involve fairness or harm to people, since no personal data is involved.

---

## Distribution

### How is the dataset available?

Through the project's public GitHub repository, alongside the notebooks that produced it.
`data/initial/` holds the supplied seed arrays, `data/accumulated/` holds the cumulative
per-round record, and `notebooks/Final/function<1-8>/` holds the authoritative analysis for
each function including its full progress log.

### Licence and terms of use

No licence file is currently present in the repository, which means that by default all
rights in the code and the query history are reserved. The intention is to release the
participant's own work — the notebooks, the analysis and the query history — for academic
review and reuse; adding an explicit licence is a pending task.

The initial seed data and the objective functions themselves are the property of the course
platform. Redistribution of those is subject to the platform's terms and is not covered by
any licence the participant adds.

### When is it available?

From the point of repository publication, submitted as part of the capstone deliverables.

---

## Maintenance

### Who maintains it?

The capstone participant, through the repository.

### How will it be updated?

It will not. All thirteen rounds are complete and the dataset is frozen. The only changes
anticipated are corrections: specifically, resolution of the round-3 attribution described
above, which would alter two trajectories but no final result, and which will be recorded
here and in the affected notebooks rather than applied silently.

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

Re-running a notebook top to bottom regenerates every number it reports. Historical rounds
are anchored to the data available at the time rather than to the full dataset — Function
1's rounds 1 to 12 run on a 21-point snapshot, its round 13 decision on 22 points, and its
closing section on all 23 — so the queries that were actually submitted still reproduce
after later observations were appended.

**Two known reproducibility caveats.**

Functions 7 and 8 both observed that proposals differ across machines in the fifth and sixth
decimal places. This traces to multi-start L-BFGS-B landing in different basins under
different BLAS builds, not to a different optimum. Function 8 measured the discrepancy at up
to 8e-5, roughly 250 times smaller than its model σ of 0.021 — immaterial to results, but
enough to confuse an exact-match check.

`requirements.txt` is unpinned and omits `seaborn`, which all eight notebooks import.
Anyone attempting bitwise reproduction should pin exact versions, since scikit-learn's
optimiser defaults have changed across minor releases.

### What a reviewer would need

The `.npy` seed files, `plotting_utils.py`, the final notebooks, and the written
reflections. The last of these is not optional. Several decisions reflect judgement about
risk and remaining budget rather than any formula: Function 3's delta and beta schedules,
Function 1's `xi_multiplier` schedule (3.0 → 2.0 → 1.5 → 0.25 → 0.05 → 0.02) and its later
rule of maximising Expected Improvement subject to at least a 50% chance of beating the
incumbent, Function 8's gain-to-σ threshold, Function 4's advance commitment to a probe
distance band of 0.05 to 0.12, and Function 7's withdrawal of its own move-type rule in the
final round. The code shows *what* was queried; only the prose explains why that direction
was chosen over the alternatives, and in at least one case why an earlier conclusion was
abandoned.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., &
  Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92.
  https://doi.org/10.1145/3458723
- Imperial College London (2026). *Professional Certificate in Machine Learning and
  Artificial Intelligence — Black-Box Optimisation Capstone*. Initial observations and
  per-round query responses supplied by the capstone portal.

See the [project README](../README.md) for the method, results and limitations, and
[`model_cards.md`](model_cards.md) for per-function model documentation.
