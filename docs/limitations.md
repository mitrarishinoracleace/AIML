# Limitations

**Imperial College London — Professional Certificate in Machine Learning and Artificial Intelligence**

| | |
|---|---|
| **Author** | Rishin Mitra |
| **Version** | 1.0 |
| **Date** | 31 October 2026 |
| **Repository** | [mitrarishinoracleace/AIML](https://github.com/mitrarishinoracleace/AIML) |
| **Status** | Complete — 13 of 13 rounds submitted for all eight functions |

---

This document states what the results do and do not support. It is deliberately more
restrictive than the results table, because the evidence is narrower than the headline
numbers suggest.

---

## 1. The central claim, stated precisely

**What can be claimed:** for each of eight functions, the best input/output pair found within
a budget of thirteen queries, starting from a supplied initial sample, together with a
documented account of how each query was chosen.

**What cannot be claimed:** that any of these is the function's global maximum — with one
exception. Function 8's generating function was recovered in closed form (additively
separable quadratic, 17 parameters, residuals ≤ 1e-15 both in-sample across 52 observations
and out-of-sample), so its optimum of exactly 10.000000 is verifiable rather than merely
observed. Every other result is the best of the regions actually explored.

This distinction matters most where the number is most impressive. Function 5 reached
8662.405 at a domain vertex, which is 99.999% of the reference best — but that is one vertex
among sixteen in 4D, and most of the interior was never queried. Function 1 improved by about
5.5e10× — from a model fitted to 23 points in a space whose output spans roughly 121 orders of
magnitude.

---

## 2. Coverage is uneven by construction

The methods that produced the improvements — trust regions, frozen coordinates, local
refinement — are precisely what prevented a wider search. This is not an oversight to be
corrected; it is the trade the budget forced. But it bounds what the data can say.

Measured at close, as the span each dimension was given by the thirteen queries:

| Function | Dimensions moved across ≤ 1/5 of their range | Queries within 0.1 of the final best |
|---|---|---|
| 4 | `x2` (0.088) | 10 of 13 |
| 6 | `x4` (0.161) | 1 of 13 |
| 7 | `x1` (0.067), `x4` (0.199), `x5` (0.058), `x6` (0.096) | 1 of 13 |
| 8 | `x1` (0.174), `x2` (0.157), `x3` (0.106), `x4` (0.163) | 1 of 13 |

Initial samples had covered roughly 0.72 to 0.98 of each range, so the queries explored far
less than the data they were added to.

**Function 7 is the clearest case and should be read as a local result.** Four of its six
dimensions were probed across a fifth or less of their range, every high-performing
observation sits in a single narrow slice, and the controlled single-variable method that
produced all six records is exactly what kept it there.

**Function 8's apparent full span in `x8` is misleading.** It reflects a binary toggle between
0 and 0.999999 across several rounds rather than a sweep, and `x8` was frozen entirely in
others.

---

## 3. The data describes the sampling policy as much as the objective

Every query from round 2 onward was chosen using a model fitted to all preceding points, so
the observations are sequentially dependent rather than independently drawn. The consequence
is measurable: Function 8's round 11 clustering analysis found exactly **one** dense region in
the combined dataset, composed entirely of model-guided queries, with all forty supplied
initial points classified as noise.

Three specific traps follow, each documented in a notebook:

- **Correlation confounded by early queries.** Function 6's milk and sugar both correlate
  negatively with output across all observations, but the correlation is driven by early
  corner queries that paired low sugar and milk with extreme butter and eggs. Taking it at
  face value would have argued against a query the ARD kernel correctly supported.
- **A long ARD length-scale read as irrelevance.** Function 8 froze `x8` for four rounds on
  this reading. A controlled test refuted it at −4.50σ. A long length-scale means the
  likelihood found no *evidence* of importance, which conflates "this does not matter" with
  "you never showed me the contrast."
- **A rule derived from a confounded sample.** Function 7's move-type bias rule rested on three
  failed queries in which two candidate causes were perfectly confounded, and the pooled
  statistic concealed the one episode that would have discriminated. It survived five rounds
  before being withdrawn in round 13.

**Consequence:** feature-importance, clustering and correlation results from this dataset
describe where the search went. They should not be read as properties of the objective
functions.

---

## 4. Reported uncertainty is not reliable everywhere

Calibration was checked every round, and it was not uniformly good:

| Function | Measure | Value | Target |
|---|---|---|---|
| 6 | Leave-one-out &#124;z&#124; > 2 | ~1 point in 5 | ~1 in 20 |
| 6 | Worst forward z | −4.0 | within ±2 |
| 5 | Leave-one-out coverage (round 10) | 82.8% | 95% |
| 5 | Leave-one-out coverage (round 8) | 75.0% | 95% |
| 2 | Round 11 forward error | −5.7σ | within ±2σ |

Where the posterior is overconfident, every Expected Improvement and UCB value computed from
it is optimistic, and the search is drawn to regions whose promise is overstated. Function 6's
final round used a 30-member bootstrap ensemble rather than its single best-fitting model for
exactly this reason.

**The sharper problem is that good calibration is not sufficient.** By round 11 Function 8's
GP was well calibrated — its prediction missed by 0.00006, or −0.04σ, the most accurate of the
project — and still systematically blind, because a stationary kernel with a long length-scale
represents a weak additive effect as no effect at all. A well-calibrated value estimate inside
a misspecified model is exactly as dangerous as a badly calibrated one, and considerably more
persuasive.

---

## 5. Single observations and the limits of one query per round

No point was ever queried twice. Spending a round on reproducibility was never affordable, and
the consequences are real:

- **Observed bests cannot be separated from favourable draws except by model fit.** Function 3
  is the clearest case: its best observed value of −0.003116 sits against a GP posterior mean
  of about −0.0102 at neighbouring points and a fitted noise standard deviation of 0.008. Its
  late improvements of 0.003 to 0.007 are *smaller than that noise*. "Exploiting the best
  observed point" can mean exploiting a lucky draw, which is why its final query used a smooth
  fit over five ridge points rather than repeating the best observation.
- **An anomaly cannot be distinguished from a real narrow feature.** This requires repeated
  queries at the same point. Function 4 spent roughly five rounds unable to resolve the
  question.
- **Three episodes cannot separate two competing explanations.** Function 7's withdrawn rule
  is the warning. A confident correction derived from a confounded sample of three is worse
  than no correction at all.

Mitigating this: the functions appear deterministic. Function 5's output is exactly linear in
`x2` along the corner ridge (R² = 0.999995) across queries taken weeks apart, and Function 8's
recovered closed form reproduces every observation to 1e-15. Where that holds, apparent noise
is model misfit rather than measurement error — which makes the single-observation limitation
less severe than it would otherwise be, but it was not verified for every function.

---

## 6. Known data defects

### Unresolved round-3 attribution between Functions 4 and 5

In `data/accumulated/week3/outputs.txt`, six of the eight values match their corresponding
notebook exactly. Two are shifted by one position against each other: 2602.253793 sits in
Function 5's slot but was by mistake recorded in Function 4 notebook(due to oversight), and −1.359780 sits in Function 6's
slot but was used by both Function 5 and Function 6, again due to oversight. 

The portal ordering is the likelier reading — 2602 lies inside Function 5's observed range and
three orders of magnitude outside Function 4's, 0.557221 sits squarely in Function 4's eventual
plateau, and Function 3 independently found and corrected this same off-by-one in round 5.

**No final best were affected**; Function 4's record and Function 5's were both set in round 13.
Two intermediate narratives inherit the ambiguity: Function 4's chase of an extreme value
across rounds 4 to 9, and the signed-log transform Function 5 introduced to handle a value
below −1. Full detail is in [`datasheet.md`](datasheet.md).

### Corrected defects

- **Function 6, round 11** — the data cell held a copy of round 1's point and output instead of
  the real round 11 result, which means round 12's submitted query was computed on a dataset
  containing a wrong row. Found and corrected in round 13; the round 12 section is frozen to
  the data as of that decision and re-run on corrected data for transparency.
- **Function 3, round 3** — output corrected at source in round 5.

---

## 7. Comparisons against the reference best are not like-for-like

The reference best for each function is quoted in the results tables, but it is not documented
whether those figures were produced under the same thirteen-query budget, from the same initial
sample, or at all comparably. Statements such as "68.6% of the reference" should be read as
positional information, not as a measured shortfall against a fair baseline.

---

## 8. Reproducibility caveats

- **Cross-machine variation.** Proposals differ in the fifth and sixth decimal places across
  machines, traced to multi-start L-BFGS-B landing in different basins under different BLAS
  builds. Function 8 measured the discrepancy at up to 8e-5, roughly 250 times smaller than its
  model σ of 0.021 — immaterial to results, but enough to defeat an exact-match check. Function
  4 added a kernel tie-break rule in round 12 specifically because near-tied leave-one-out
  scores were making kernel selection machine-dependent.
- **Unpinned dependencies.** `requirements.txt` specifies no versions and omits `seaborn`, which
  all eight notebooks import. scikit-learn's optimiser defaults have changed across minor
  releases, so bitwise reproduction requires pinning.
- **No licence file.** The repository currently carries no licence, so by default all rights are
  reserved and the code is not formally reusable.

---

## 9. Scope of the method itself

Independent of this dataset, the approach as implemented has known boundaries:

- **Exact GP inference is O(n³).** A few hundred more points before sparse or inducing-point
  approximations become necessary.
- **Gradient-based acquisition maximisation under-proposes domain vertices.** Function 5's final
  result sat at one, against a GP prediction roughly seventy times lower. Vertex candidates must
  be added to the candidate set explicitly.
- **Stationary kernels cannot represent effects that are sharp locally and absent on average**,
  nor objectives whose output scale varies by many orders of magnitude without a transform.
  Function 1 lost seven rounds to the second of these.
- **Continuous surrogates are wrong for categorical inputs.** If Function 8's `x8` encodes
  optimiser type, as its brief suggests, there is no interior optimum to find and the search
  slides to whichever endpoint is marginally favoured.
- **One-factor-at-a-time buys interpretability by sacrificing the ability to detect
  interactions** — which is the method that produced most of the late records here.

---

## 10. What would most change these conclusions

Listed so that a reader can judge how fragile the results are:

1. **A standing model-class comparison** — fit an additive model, a low-order polynomial and a
   full GP, compare leave-one-out error every round. Near-zero cost, and it would have solved
   Function 8 in round 1 rather than round 13.
2. **Repeated measurement at a single point**, once, in any function. It would settle the
   determinism assumption that several conclusions rest on.
3. **Resolution of the round-3 attribution** with the course platform.
4. **A wider initial design in the dimensions later frozen**, particularly Functions 7 and 8.
   The local results would stand; the global claims would become assessable.
5. **A formal cross-machine reproducibility test** to bound the BLAS-related variation rather
   than merely note it.

---

See [`methodology.md`](methodology.md) for how queries were produced,
[`strategy.md`](strategy.md) for the per-function narrative,
[`decision_log.md`](decision_log.md) for the dated record,
[`datasheet.md`](datasheet.md) for the data and
[`model_cards.md`](model_cards.md) for model details.

**Note:** Document has been refined using AI
