# Black-Box Optimisation (BBO) Capstone

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 2.0 (final submission) |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |

---

## Contents

1. [Non-Technical Explanation](#1-non-technical-explanation)
2. [Project Overview](#2-project-overview)
3. [Data](#3-data)
4. [Model](#4-model)
5. [Hyperparameter Optimisation](#5-hyperparameter-optimisation)
6. [Results](#6-results)
7. [Limitations](#7-limitations)
8. [Repository Structure](#8-repository-structure)
9. [Reproducing the Results](#9-reproducing-the-results)
10. [Documentation](#10-documentation)
11. [References](#11-references)
12. [Licence and Contact](#12-licence-and-contact)

---

## 1. Non-Technical Explanation

Imagine eight sealed machines. Each one takes a handful of dials, set anywhere between 0 and 1, and returns a single score. You cannot open the machine, you cannot see the formula inside it, and you are allowed to turn the dials and read the score **once per machine per week**. Your job is to find the dial settings that give the highest score.

That is this project. The eight machines stand in for real problems: locating a contamination source, choosing a drug compound mix with the fewest side effects, tuning a warehouse-allocation model, maximising the yield of a chemical process, perfecting a recipe, and tuning the hyperparameters of two machine learning models.

With only thirteen readings per machine, guessing is hopeless. Instead, each week a statistical model learns from every reading taken so far, predicts what the score would be anywhere in the space **and how unsure it is about that prediction**, and uses both numbers to choose where to look next. Over thirteen rounds all eight functions improved on their starting point, one of them by a factor of about 55 billion, and one was solved exactly.

The more useful result is not the scores. It is the record of *why* each query was chosen, which predictions failed, and what each failure taught — because that is the part that transfers to any expensive experiment.

---

## 2. Project Overview

### Purpose

This repository documents a thirteen-week black-box optimisation (BBO) exercise over eight unknown objective functions with input dimensions ranging from two to eight. Each function is a sealed system: its analytical form and gradients are unavailable, and the only way to probe it is to submit an input point and observe the returned scalar. With a strict limit of one query per function per week, every evaluation had to be justified in advance and audited afterwards.

That constraint is the point of the exercise. When evaluations are cheap, grid or random search will do. When a single evaluation means a clinical trial, a materials synthesis run, or a full model training job, choosing where to evaluate becomes a consequential decision that has to be defended with evidence and documented so it can be scrutinised and reproduced.

### Real-World Relevance

Black-box optimisation is the standard framing for hyperparameter tuning, drug and materials discovery, industrial process optimisation and simulation-based design — settings where the evaluation budget, not the model, is the binding constraint.

Four lessons stand out from the thirteen rounds, each traceable to a specific round in a specific notebook:

- **Model confidence is not evidence.** Function 8's relevance estimates and calibration diagnostics consistently dismissed one input until a controlled single-variable test contradicted them at −4.50σ. The posterior was *well calibrated* and still structurally blind: several agreeing diagnostics can share one blind spot.
- **Exploitation is only as good as the model being exploited.** Function 1 spent seven rounds refining a neighbourhood and never improved on its initial best, because the surrogate was fitted on raw outputs whose spread was twelve orders of magnitude wider than the basin it was meant to resolve. Re-modelling the same data in log-space, with no new observations, found a 55-billion-fold improvement in two rounds.
- **Sampling shapes what you discover.** Model-guided queries increasingly reflect existing assumptions. A clustering analysis of Function 8 in Round 11 found exactly one dense region, composed entirely of my own queries, with all forty initial points classified as noise. Feature-importance and cluster structure can describe the sampling policy rather than the objective.
- **Calibration matters more than fit alone.** Raising Function 8's noise floor from 1e-8 to 1e-4 worsened the in-sample fit and put five of the next six predictions inside 1σ. Under a fixed budget, uncertainty you can act on beats a tighter fit that invites misplaced confidence.

### Headline Result

| Metric | Value |
|---|---|
| Functions optimised | 8 (2D to 8D) |
| Rounds | 13 per function (104 queries in total) |
| Observations at close | 279 (175 initial + 104 queried) |
| Functions improved on their initial best | **8 of 8** |
| Best found in the final round | 6 of 8 (F1, F4, F5, F6, F7, F8) |
| Functions matching or exceeding the published reference best | 4 of 8 exceeded or matched exactly (F1, F2, F3, F8); F5 within 0.001% |
| Global optimum provably attained | Function 8 (function recovered in closed form; residuals ≤ 1e-15) |

---

## 3. Data

### Source

No external dataset is used. All data is generated by querying the capstone portal, and arrives in two parts per function.

1. **Initial observations** — supplied by the portal at the start of the exercise, as NumPy arrays:
   - `data/initial/function<1-8>/initial_inputs.npy` — shape `(n_initial, d)`
   - `data/initial/function<1-8>/initial_outputs.npy` — shape `(n_initial,)`
2. **Weekly query history** — one input/output pair added per function per round, accumulated as plain text:
   - `data/accumulated/week<1-13>/inputs.txt`
   - `data/accumulated/week<1-13>/outputs.txt`

Each weekly file holds one entry per function, in function order 1 to 8. The notebooks hold the same points inline as literal arrays, so every notebook runs standalone from its own directory without reading the `data/` tree.

### Dataset Characteristics

| Function | Dimensions | Initial points | Points at close | Initial best | Output character | Application framing |
|---|---|---|---|---|---|---|
| F1 | 2 | 10 | 23 | 7.710875e-16 | Positive, spanning ~121 orders of magnitude | Contamination / radiation source detection |
| F2 | 2 | 10 | 23 | 0.611205 | Moderate positive, noisy | Log-likelihood of a mystery model |
| F3 | 3 | 15 | 28 | −0.034835 | Negative (penalty) | Drug compound mix, minimising adverse reactions |
| F4 | 4 | 30 | 43 | −4.025542 | Negative crossing to positive, with one extreme outlier in the round-3 record | Warehouse allocation surrogate |
| F5 | 4 | 20 | 33 | 1088.86 | Large positive | Chemical process yield |
| F6 | 5 | 20 | 33 | −0.714265 | Negative (penalty) | Cake recipe scored on five ingredients |
| F7 | 6 | 30 | 43 | 1.364968 | Moderate positive | ML hyperparameter tuning |
| F8 | 8 | 40 | 53 | 9.598482 | Moderate positive, near-plateau | ML hyperparameter tuning (8 parameters) |

All inputs are bounded to `[0.000000, 0.999999]` per dimension. All outputs are unbounded scalars. The functions are **deterministic** — querying the same point twice returns the same value — which was confirmed for Function 5, where output is exactly linear in `x2` along the corner ridge (R² = 0.999995). Apparent noise is therefore model error, not measurement error, and several rounds were spent learning that distinction.

### Inputs

Each query is a vector of floats in `[0.000000, 0.999999]`, one per dimension, formatted to exactly six decimal places and submitted to the portal as a hyphen-separated string:

```
# Function 3, round 13 — three compounds
0.463877-0.612656-0.417300

# Function 8, round 13 — eight hyperparameters
0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000
```

### Outputs

Each query returns a single unbounded float, which the portal reports to full double precision. Scale varies enormously between functions, which is why per-function output transforms were needed rather than one global convention:

```
F1  round 13 : 4.240382032191716e-05      # positive, near-zero magnitudes
F5  round 13 : 8662.405001248297          # four orders of magnitude larger
F6  round 13 : -0.21355384462410984       # negative by construction
F8  round 13 : 10.0                       # exact
```

### Data Citations

- Imperial College London (2026). *Professional Certificate in Machine Learning and Artificial Intelligence — Black-Box Optimisation Capstone*. Initial observations and weekly query responses supplied by the capstone portal.
- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92. — the format followed by [`docs/datasheet.md`](docs/datasheet.md).

---

## 4. Model

### Gaussian Process Surrogate with an Acquisition Function

The core method for all eight functions is a **Gaussian Process (GP) surrogate** driving an **acquisition function**, implemented with `sklearn.gaussian_process.GaussianProcessRegressor` and optimised with multi-start L-BFGS-B from `scipy.optimize`. Given every input/output pair observed so far, the GP returns a posterior mean μ(x) and a posterior standard deviation σ(x) at any untested point; the acquisition function combines the two into a single score, and the argmax of that score is the query submitted.

### Why a GP Surrogate

The choice follows from the situation rather than from a preference for GPs:

1. **The budget is tiny.** Datasets run from 10 points in 2D to 53 points in 8D. At that size, model flexibility is worth less than knowing where the model is ignorant; a neural network would carry more parameters than data points. (A `src/surrogates/neural_net.py` alternative was written and set aside for this reason — see [`docs/model_cards.md`](docs/model_cards.md).)
2. **Uncertainty is the input to the decision, not a diagnostic.** Both acquisition functions used here need σ(x) to exist. Upper Confidence Bound scores μ(x) + βσ(x); Expected Improvement integrates over the posterior. Neither is definable without calibrated uncertainty.
3. **There are theoretical guarantees under exactly this budget.** Srinivas et al. (2010) establish no-regret bounds for GP-UCB in the sequential, bandit-style setting this exercise imposes, which is the formal reason to prefer it to random or grid search.
4. **ARD length-scales are a free relevance measure.** Automatic Relevance Determination assigns one length-scale per input; a length-scale driven to its upper bound is the kernel declaring an input irrelevant. This diagnostic found the `x3` constraint that produced Function 7's best result — and, in Function 8, produced a confidently wrong answer, which is the more instructive outcome.

### Kernel Configuration

Each round refitted the kernel and re-ran a **bake-off** between a Matérn ARD kernel and an RBF ARD kernel, selected on leave-one-out cross-validated RMSE or log-marginal likelihood. The configured candidate set and the kernel that usually won per function:

| Function | Kernel usually selected | ν | Length-scale bounds | Noise bounds (`WhiteKernel`) | Rationale |
|---|---|---|---|---|---|
| F1 | Matérn ARD | 2.5 | (0.02, 1.0) | (1e-10, 1e-2) | Near-singular basin; tight lower bound needed to resolve it at all |
| F2 | RBF ARD | — | (1e-2, 5.0) | (1e-5, 1e0) | Genuinely noisy output; the two kernels scored within ~6% of each other |
| F3 | Matérn ARD | 1.5 | (1e-2, 10) | (1e-6, 1e-1) | Matérn retained in most weeks (e.g. LOO 0.084 vs RBF 0.090) |
| F4 | Matérn ARD | 2.5 | (1e-2, 2.0) | (1e-6, 1e-1) | Jagged surface with an extreme outlier; needs a short-correlation kernel |
| F5 | Bake-off, both won rounds | 1.5 | (1e-2, 1e2) | (1e-5, 1e0) | Smooth, large space; outputs modelled under a signed-log transform |
| F6 | Matérn ARD, standardised outputs | 2.5 | (1e-2, tuned upper) | (1e-5, 1.0) | All-negative outputs; standardising prevents unexplored regions scoring above observed ones |
| F7 | Matérn ARD | 2.5 | (1e-2, 1e1) | (1e-6, 1e0) | Six dimensions, ARD central to the strategy |
| F8 | Matérn ARD | 2.5 | (1e-2, 1e2) | (1e-4, 1e0) | Noise floor deliberately *raised* from 1e-8 after Week 3's −1.94σ miss |

A `ConstantKernel` scale factor multiplies every kernel, and a `WhiteKernel` noise term is fitted rather than fixed, except where noted.

### Where the GP Was Replaced

Two functions were not solved by the GP, and saying so is part of the result.

- **Function 1** — the GP fitted on raw `y` could not see the target basin, because one −3.6e-3 reading set an output spread about twelve orders of magnitude wider than the basin's own. From Round 12 the GP was abandoned for a **parametric two-source log-magnitude forward model** fitted on all observations, with Expected Improvement computed by Monte Carlo over an unknown sign-flip radius. It predicted the final query's outcome to within a factor of 0.6 — the best out-of-sample prediction any model produced in this project.
- **Function 8** — after twelve rounds of kernel and acquisition tuning on a plateau, the function was identified as **additively separable quadratic with 17 parameters** and solved in closed form. The recovered model reproduces all 52 observations and holds out-of-sample to residuals ≤ 1e-15, and its analytic argmax returned exactly 10.000000. Seventeen observations were sufficient to identify the function; forty were supplied before the first query. The missing step was never more data or a better acquisition function — it was one round of *structural* model selection across model families rather than hyperparameter tuning within one family.

Supporting models used alongside the GP: a ridge-quadratic fit along a learned diagonal (F3), a local ARD GP fitted only to the plateau with the outlier excluded (F4, LOO-CV RMSE 0.033 against 0.54 for a flat-plateau baseline), and a 30-model bootstrap ensemble for bag-averaged Expected Improvement (F6).

### Model References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. — kernel and length-scale reasoning; the ARD treatment is how Function 8's `x8` was identified as carrying no signal, its length-scale having run to 232 against neighbours of 3 to 5.
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient Global Optimization of Expensive Black-Box Functions. *Journal of Global Optimization*, 13(4), 455–492. — the fit-surrogate-then-select loop, and the reason Expected Improvement collapses to zero once the model believes nothing can beat the incumbent. Function 8 hit that failure exactly in Round 5, which is what ruled EI out and left UCB.
- Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural Information Processing Systems*, 25. — Matérn 5/2 as the default over RBF on rough response surfaces, which the weekly cross-validation generally bore out, and ARD length-scales as a relevance measure.
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *FAccT 2019*, 220–229. — the format followed by [`docs/model_cards.md`](docs/model_cards.md).

---

## 5. Hyperparameter Optimisation

The consequential hyperparameter in this project is not a kernel setting. It is the **acquisition function and its exploration weight**, because that is what decides where the single weekly query goes.

### Acquisition Strategy and Exploration Schedule

| Function | Acquisition at close | Exploration weight | Search region at close | Rationale |
|---|---|---|---|---|
| F1 | Monte-Carlo EI on a two-source parametric model | β ≈ 0 (pure exploitation) | Ray from the Round 12 point toward the fitted strong source | Rule: maximise EI subject to P(beat best) ≥ 50%, rather than unconstrained EI at ~22% |
| F2 | Space-filling tie-break | n/a | Largest unsampled `x2` interval inside the proven `x1 ≈ 0.65–0.76` band | β wound 2.5 → 0.15; own-data models were indifferent between candidates, so the query filled the largest evidence gap |
| F3 | Ridge-quadratic peak + GP for `compound_3` | β = 0 | Along the learned diagonal (PC1 explains 57% of sampled input variance) | β 2.5 → 0.1–0.3 → 0; a smooth fit over five points rather than repeating a possibly lucky best observation |
| F4 | Mean + βσ, exploration probe | β = 2.0 | 0.05–0.12 from the best confirmed point, ≥ 0.05 from every queried point, outlier excluded | Plateau refinement had stopped paying; probe rule fixed in writing before the result was known |
| F5 | Hard vertex probe | n/a | The unexplored far vertex `(0.999999)⁴` | GP posterior there was uninformed (σ in log-space ≈ 2.05); the acquisition optimiser structurally favours interior points |
| F6 | Bag-averaged EI over 30 bootstrap GPs + full fit | n/a | Centre of the top-300 candidates, between the surrogate mode and the Round 12 region | Single fit is overconfident (LOO \|z\| > 2 for ~1 in 5 points), so the ensemble hedges rather than trusting one mode |
| F7 | GP optimum with `x1` held in its proven band | n/a | Six coordinates moved together after a prior bias rule was withdrawn | β 2.0 → 1.2 → 0.8 abandoned at Round 8 for controlled single-variable steps |
| F8 | Analytic argmax of the recovered function | n/a | Whole domain | No acquisition function needed once the closed form was identified |

### UCB β Schedule — Function 8

β was the entire strategy for the first four rounds, and the record of how it moved is more informative than any single value:

| Round | β | Outcome |
|---|---|---|
| 1–4 | 2.0, 1.5, 1.0, 0.7 | Two improvements, then a stall |
| 5 | **3.0** (raised) | EI had collapsed to identically zero, so β was raised and a probe sent 0.23 from the basin. Returned 0.065 *below* the incumbent — a loss in value that confirmed the basin and validated a calibration fix in the same round |
| 6 | 0.5 | Returned 0.0039 below the incumbent at −0.44σ; plateau confirmed, and improvements were now smaller than the model's own σ |
| 8 onward | n/a | Switched from β tuning to controlled single-variable steps, which produced three of the five improvements |

### Gain-to-σ Decision Rule — Function 8

Rather than tuning β further, risk was formalised: a direction earns a query only when its predicted gain clears the model's own uncertainty. The rule discriminated cleanly:

| Gain ÷ σ | 1.45 | 0.87 | 0.81 | 0.44 | 0.29 | 0.11 |
|---|---|---|---|---|---|---|
| Outcome | New best | New best | New best | Failed | Failed | Failed |

Round 12 was forecast as likely to fail at a ratio of 0.44, and did, missing by 0.00012.

### Noise Floor Constraints

The fitted noise term was the single most consequential kernel hyperparameter, because it sets how confident the posterior is allowed to be:

- **Function 8** — the learned noise level kept collapsing to its lower bound, leaving σ at the incumbent around 1e-4; the GP was interpolating 44 points in 8D rather than regularising them. Raising the floor from 1e-8 to 1e-4 worsened the in-sample fit and put five of the next six predictions inside 1σ.
- **Function 2** — a near-zero noise estimate inflated confidence and contributed to two misses when the strategy pushed toward pure exploitation in Round 10. The fitted noise later rose sevenfold after a −5.7σ miss.
- **Function 5** — a GP fitted with near-zero noise on a function that turns out to be deterministic is a signal that small differences are *signal*, not noise. Rounds 11 and 12 mistook a ramp for a plateau for exactly this reason.

### Validation Protocol

Every round ran the same checks before a query was submitted, which is what made the failures legible rather than merely disappointing:

- **Leave-one-out cross-validation** — RMSE, mean |z| and interval coverage against the nominal 95%.
- **Forward (out-of-sample) test** — refit on all data *before* round *k* and compare the prediction for round *k* against what was actually returned. This produced the per-round prediction-error record that drove most strategy changes.
- **Kernel bake-off** — Matérn versus RBF, re-run every round rather than fixed once.
- **Calibration table** — predicted ± σ against actual for every round, carried forward across the project.
- **Fixed seed** (42) and a single reusable `bayesian_optimization_step()` per notebook, so each round's decision cell reproduces the point that was actually submitted even after later data is appended.

---

## 6. Results

### Final Record — All Eight Functions

| Function | Initial best | Final best | Round | Improvement | Reference best | Versus reference |
|---|---|---|---|---|---|---|
| F1 | 7.710875e-16 | **4.240382e-05** | 13 | ~5.5e10× | 3.50e-06 | **~12× higher** |
| F2 | 0.611205 | **0.711209** | 9 | +16.4% | 0.668242 | **+6.4%** |
| F3 | −0.034835 | **−0.003116** | 11 | ~11× smaller penalty | −7.172e-03 | **~2.3× better** |
| F4 | −4.025542 | **0.590944** | 13 | +4.617 (sign change) | 0.730826 | 81% of it |
| F5 | 1088.86 | **8662.405001** | 13 | ~8.0× | 8662.4825 | **99.999% of it** |
| F6 | −0.714265 | **−0.213554** | 13 | penalty ~70% smaller | −0.068174 | penalty ~3.1× larger |
| F7 | 1.364968 | **2.240937** | 13 | +64.2% | 3.267413 | 68.6% of it |
| F8 | 9.598482 | **10.000000** | 13 | +4.18% | 10.000000 | **matched exactly** |

**All eight functions improved on their initial best.** Six of the eight set their record in the final round, which suggests the trajectories had not flattened.

### Convergence

Best-so-far by round, computed from the portal files in `data/accumulated/`:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 (×1e-16) | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 7.71 | 4.7e9 | **4.2e11** |
| F2 | 0.611 | 0.611 | 0.611 | 0.611 | 0.611 | 0.619 | 0.620 | 0.620 | **0.711** | 0.711 | 0.711 | 0.711 | 0.711 |
| F3 | −0.0348 | −0.0348 | −0.0348 | −0.0348 | −0.0325 | −0.0325 | −0.0325 | −0.0125 | −0.0040 | −0.0040 | **−0.0031** | −0.0031 | −0.0031 |
| F4 | −0.066 | −0.066 | 0.557 | 0.557 | 0.561 | 0.561 | 0.561 | 0.577 | 0.577 | 0.577 | 0.577 | 0.577 | **0.591** |
| F5 | 1089 | 1909 | 1909 | 1909 | 1909 | 1909 | 1915 | 1915 | 1915 | 1915 | 1915 | 1915 | **8662** |
| F6 | −0.711 | −0.711 | −0.711 | −0.711 | −0.711 | −0.465 | −0.465 | −0.465 | −0.465 | −0.402 | −0.402 | −0.362 | **−0.214** |
| F7 | 1.365 | 1.365 | 1.365 | 1.365 | 1.597 | 1.597 | 1.597 | 1.647 | 1.698 | 1.869 | 1.873 | 1.899 | **2.241** |
| F8 | 9.888 | 9.939 | 9.939 | 9.974 | 9.974 | 9.974 | 9.974 | 9.976 | 9.976 | 9.981 | 9.983 | 9.983 | **10.000** |

### Key Findings

**Function 1 — re-modelling beat re-sampling.** Eleven rounds passed without improving on the initial best, seven of them (5 to 11) spent on local refinement around the incumbent. It produced nothing, because the GP was fitted on raw outputs spanning 121 orders of magnitude and could not resolve a basin whose own spread was 3e-16. Modelling the same data in log-space — no new observations — identified a two-source structure, placed the stronger source about 5σ beyond the sampled region, and converted that into a ~5.5e10× improvement across the last two rounds. **The dominant failure mode was the model class, not the search.**

**Function 5 — the optimum was at a vertex the acquisition optimiser could not propose.** Three of four inputs sat on their boundary in the region the search committed to from Round 2, and moving any of them by 0.01–0.05 cost hundreds of yield points. A deliberate probe of the unexplored far vertex `(0.999999)⁴` in the final round returned 8662.405 against a GP prediction of about 124 — the surrogate was uninformed there, not wrong. Gradient-based acquisition maximisers favour interior points; vertices need to be proposed explicitly.

**Function 8 — the answer was over-determined before the first query.** The function is additively separable quadratic with 17 parameters, identifiable from 17 observations; 40 were supplied up front. Twelve rounds of kernel, noise-floor and acquisition tuning missed it because the search ran inside a model family that could not represent it. A standing cheap diagnostic — fit an additive model, a low-order polynomial and a full GP, and compare LOO error every round — would have flagged this immediately. An LOO-RMSE at machine precision is not a tuning success; it means the generating function has been recovered.

**Function 7 — controlled single-variable steps worked, and that is also what limited them.** Abandoning the annealed β schedule at Round 8 in favour of moving one coordinate at a time produced six records and a +64.2% improvement, with each gain cleanly attributable (for example +0.588 from `x4` alone, established by a controlled pair). The cost is honest: four of six dimensions were probed across 20% or less of their range, so this is a well-characterised **local** optimum. In the final round a bias rule derived from three failed queries was tested, found to rest on confounded evidence, and withdrawn — which reversed the final decision and produced the project's largest single-round gain (+0.342).

**Function 4 — one extreme reward had to be quarantined, and that was the right call for the wrong reason.** A round-3 output of 2602.25 — against a prior range of −32.6 to −0.07 — never reproduced across eight nearby probes. Excluding it from the plateau model was correct: treating it as ordinary would have corrupted every subsequent value estimate, exactly as one extreme reward skews an RL agent's value function. The portal files subsequently show that value recorded against Function 5 rather than Function 4 (attribution under review), which makes it a data-alignment artefact rather than a narrow peak. Either way the confirmed trajectory moved from −4.03 to +0.59, and the final round's exploration probe — rule fixed in writing in advance, forecast 0.364 ± 0.386 — returned 0.591 and became the new confirmed best.

**Function 3 — exploitation is not noise-free either.** The best observed value (−0.003116) sits against a GP posterior mean of about −0.0102 at neighbouring points and a fitted noise standard deviation of 0.008, so part of it is a favourable draw. Late improvements of 0.003–0.007 were smaller than that noise, which is why the final query used a smooth fit over five ridge points rather than repeating the best observation.

### Strategy Evolution

| Phase | Rounds | Approach | Outcome |
|---|---|---|---|
| Scheduled annealing | 1–4 | GP-UCB/EI with β wound down on a calendar (2.5 → 0.15 on F2; 2.0 → 0.7 on F8) | Boundary-hugging proposals; F3's five corner queries scored −0.050 to −0.082; F7 beat nothing in its initial data |
| Diagnostic turn | 5–8 | Calibration residuals and forward tests treated as the primary signal; noise floors raised; EI collapse diagnosed | F8's noise fix (five of next six inside 1σ); F3's first real improvement (−0.0125 vs −0.0325); F6's manual hypothesis beat the automated search |
| Controlled single-variable steps | 8–11 | One coordinate moved at a time, with the rest frozen | F7's and F8's records became individually attributable; F7 LOO mean \|z\| 1.56 → 0.89 locally |
| Structural model change | 12–13 | Model *class* replaced rather than retuned: log-space two-source model (F1), closed-form solve (F8), ridge-quadratic (F3), bootstrap ensemble (F6), vertex probe (F5) | Six of eight records set in the final round; F8 solved exactly; F1 improved ~5.5e10× over its initial best |

---

## 7. Limitations

Stated plainly, because the evidence supports a narrower claim than the results table alone suggests.

- **These are the best values found, not proven global optima** — with one exception. Function 8's recovered closed form makes its optimum verifiable; every other result is the best of the regions actually explored. Much of Function 5's 4D space and Function 7's 6D space was never queried at all.
- **Coverage is uneven by construction.** The strategies that produced the improvements — trust regions, frozen coordinates, local refinement — are precisely what prevented a wider search. Function 7 probed four of six dimensions across 20% or less of their range; every high-performing Function 8 observation sits in one dense cluster made entirely of model-guided queries.
- **Some reported uncertainty is too narrow.** Function 6's leave-one-out |z| exceeds 2 for roughly one point in five, and its forward z reached −4.0. Function 5's LOO coverage fell to 75% near the corner against a nominal 95%. Where the posterior is overconfident, the EI and UCB values computed from it are optimistic.
- **Small samples cannot separate competing explanations.** Function 7's withdrawn bias rule is the cautionary case: two candidate features were perfectly confounded across three failed queries, and the pooled statistic concealed the single episode that would have discriminated between them.
- **One observation per function per round.** No repeated measurement at any point, so observed values cannot be separated from favourable draws except by model fit (Function 3 is the clearest case). The functions appear deterministic, which makes this less severe than it would otherwise be, but it was not verified for every function.
- **Rankings against the reference best are not like-for-like** unless the reference was produced under the same 13-query budget, which is not documented.
- **One round-3 output has an unresolved attribution.** In `data/accumulated/week3/outputs.txt`, six of the eight values match the corresponding notebook exactly. The remaining two are shifted by one position against each other: the value 2602.253793 sits in Function 5's slot but was used in the Function 4 notebook, and −1.359780 sits in Function 6's slot but was also used in Function 5's. This is the same off-by-one that Function 3 identified and corrected in round 5. The portal ordering is the more likely reading, since 2602 lies inside Function 5's observed range and three orders of magnitude outside Function 4's, while 0.557221 sits squarely in Function 4's eventual plateau. **No final best is affected** — Function 4's record (0.590944) and Function 5's (8662.405001) were both set in round 13 — but Function 4's rounds 4 to 9 were spent probing a value that may not belong to it, and Function 5's best-so-far trace between rounds 3 and 12 is in question. Pending resolution, the convergence table follows the portal files and the per-function narratives are flagged where they differ.

---

## 8. Repository Structure

```
AIML/
├── README.md                        # this file
├── requirements.txt                 # Python dependencies
├── configs/
│   └── optimisation.yaml            # seed, surrogate, acquisition and search defaults
├── data/
│   ├── initial/function<1-8>/       # initial_inputs.npy, initial_outputs.npy
│   └── accumulated/week<1-13>/      # inputs.txt, outputs.txt (one entry per function)
├── docs/
│   ├── datasheet.md                 # dataset documentation (Gebru et al. format)
│   ├── model_cards.md               # model documentation (Mitchell et al. format)
│   ├── strategy.md                  # per-function strategy narrative
│   ├── methodology.md               # method and validation protocol
│   ├── decision_log.md              # one entry per strategy decision
│   └── limitations.md               # scope and caveats
├── notebooks/
│   ├── Final/function<1-8>/         # FINAL submission notebooks — start here
│   │   ├── Function<N>.ipynb        # full 13-round analysis, self-contained
│   │   ├── initial_inputs.npy
│   │   ├── initial_outputs.npy
│   │   └── plotting_utils.py
│   └── Week <1-13>/                 # per-round snapshots, kept as an audit trail
└── src/
    ├── pipeline.py                  # end-to-end round pipeline
    ├── optimizer.py                 # acquisition maximisation (multi-start L-BFGS-B)
    ├── search_space.py              # bounds and candidate generation
    ├── candidate_selection.py       # final query selection and tie-breaks
    ├── acquisition/ucb.py           # Upper Confidence Bound
    ├── surrogates/gp.py             # Gaussian Process surrogate
    ├── surrogates/neural_net.py     # alternative surrogate (evaluated, not used)
    ├── search/local_search.py       # trust-region / local box search
    └── diagnostics/                 # gradient and SVM-based structure checks
```

`notebooks/Final/function<1-8>/` holds the authoritative version of each function. Each notebook is self-contained — it carries its own copy of the initial `.npy` files and `plotting_utils.py`, and holds the weekly observations inline — so it runs from its own directory with no repository-relative paths.

---

## 9. Reproducing the Results

### Environment

```bash
git clone https://github.com/mitrarishinoracleace/AIML.git
cd AIML
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10 or later. The notebooks import `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` and `seaborn`, plus the local `plotting_utils.py` that ships beside each one.

> **Note:** `seaborn` is imported by all eight notebooks but is missing from `requirements.txt`, so a clean install will fail on the first plotting cell. Install it alongside the pinned set (`pip install seaborn`) or add it to the file.

### Running a Function

```bash
jupyter lab notebooks/Final/function8/Function8.ipynb
```

Run the notebook top to bottom. Each round's decision cells are anchored to the data that existed *at the time* — Function 1's rounds 1–12 run on a 21-point snapshot, its round 13 decision on 22 points, and the final section on all 23 — so historical queries reproduce exactly rather than being silently recomputed against the full dataset.

### Determinism

The random seed is fixed at 42 in `configs/optimisation.yaml` and set in each notebook. Kernel hyperparameter fitting uses multiple restarts from fixed starting points, and acquisition maximisation uses 20 L-BFGS-B restarts from a fixed Latin-hypercube design. Re-running a notebook reproduces every submitted query.

> **Reproducibility note:** `requirements.txt` is currently unpinned. Pinning exact versions (`pip freeze > requirements.lock`) would make the GP hyperparameter fits byte-reproducible across environments, since scikit-learn's optimiser defaults have changed between minor releases.

---

## 10. Documentation

| Document | Contents |
|---|---|
| [Datasheet](docs/datasheet.md) | Provenance, composition, collection process and intended use of the data, following Gebru et al. (2021) |
| [Model Cards](docs/model_cards.md) | Per-function model details, intended use, metrics, evaluation data and ethical considerations, following Mitchell et al. (2019) |
| [Strategy](docs/strategy.md) | Per-function strategy narrative across the thirteen rounds |
| [Methodology](docs/methodology.md) | Pipeline, kernel selection, acquisition rules and validation protocol |
| [Decision log](docs/decision_log.md) | One dated entry per strategy decision: what was decided, why, and what happened |
| [Limitations](docs/limitations.md) | Scope of the claims and known weaknesses |

Per-round reflections and the progress log for each function are held in the function notebooks under `notebooks/Final/` and in the weekly snapshots under `notebooks/Week <N>/`.

> `strategy.md`, `methodology.md`, `limitations.md` and `decision_log.md` are currently placeholders. The underlying material exists in each notebook's change log and progress log; the per-function change logs are the authoritative decision record until those files are populated.

---

## 11. References

### Bayesian optimisation

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. http://gaussianprocess.org/gpml/
2. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient Global Optimization of Expensive Black-Box Functions. *Journal of Global Optimization*, 13(4), 455–492. https://doi.org/10.1023/A:1008306431147
3. Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design. *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 1015–1022. https://arxiv.org/abs/0912.3995
4. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural Information Processing Systems*, 25, 2951–2959. https://arxiv.org/abs/1206.2944
5. Brochu, E., Cora, V. M., & de Freitas, N. (2010). A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning. arXiv:1012.2599. https://arxiv.org/abs/1012.2599
6. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the Human Out of the Loop: A Review of Bayesian Optimization. *Proceedings of the IEEE*, 104(1), 148–175. https://doi.org/10.1109/JPROC.2015.2494218

### High-dimensional and local Bayesian optimisation

7. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019). Scalable Global Optimization via Local Bayesian Optimization (TuRBO). *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1910.01739
8. Wang, Z., Hutter, F., Zoghi, M., Matheson, D., & de Freitas, N. (2016). Bayesian Optimization in a Billion Dimensions via Random Embeddings. *Journal of Artificial Intelligence Research*, 55, 361–387. https://doi.org/10.1613/jair.4806

### Reinforcement learning connections

9. Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. *Biometrika*, 25(3/4), 285–294. https://doi.org/10.2307/2332286
10. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. http://incompleteideas.net/book/the-book.html
11. Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., & Hassabis, D. (2017). Mastering the Game of Go Without Human Knowledge. *Nature*, 550(7676), 354–359. https://doi.org/10.1038/nature24270

### Documentation standards

12. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAccT)*, 220–229. https://doi.org/10.1145/3287560.3287596
13. Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92. https://doi.org/10.1145/3458723

### Software

14. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
15. Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array Programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
16. Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
17. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

---

## 12. Licence and Contact

**Licence** — no licence file is present yet. Adding one is recommended: without it, the default position is that all rights are reserved and others may not reuse the code. The MIT Licence suits coursework that is intended to be read and built on. The capstone data was supplied by Imperial College London for coursework and is included for reproducibility of the submitted work only; it is not covered by any code licence added here.

**Citation**

```bibtex
@misc{mitra2026bbo,
  author       = {Mitra, Rishin},
  title        = {Black-Box Optimisation Capstone: Thirteen-Round Bayesian
                  Optimisation of Eight Unknown Objective Functions},
  year         = {2026},
  howpublished = {\url{https://github.com/mitrarishinoracleace/AIML}},
  note         = {Imperial College London, Professional Certificate in
                  Machine Learning and Artificial Intelligence}
}
```

**Contact**

- **Author** — Rishin Mitra
- **Programme** — Professional Certificate in Machine Learning and Artificial Intelligence, Imperial College London
- **Repository** — [github.com/mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML)

**Note:** Document has been refined using AI
