# Black-Box Optimisation (BBO) Capstone

Imperial College London - Professional Certificate in Machine Learning and Artificial Intelligence

**Version:** 1.0
**Date:** 10 October 2026
**Author:** Rishin Mitra


## Section 1: Project Overview

### Purpose

This repository documents a twelve-week black-box optimisation (BBO) exercise involving eight unknown objective functions, with input dimensions ranging from two to eight. Each function is a sealed system, i.e, its analytical form and gradients are unavailable, and the only way to explore it is to submit an input point and observe the output. With a strict limit of one query per function per week, each evaluation was carefully chosen to maximise what was learnt and how effectively the search progressed.
That constraint is central to the project. When evaluations are cheap, grid or random search may suffice. When each evaluation takes days or weeks—or represents a clinical trial, materials synthesis, or full model training, executing every query becomes a consequential decision. The challenge is to choose where to evaluate next, justify that choice with evidence, and document the reasoning so it can be scrutinised and reproduced.

### Real-World Relevance

Black-box optimisation mirrors real-world challenges such as hyperparameter tuning, drug discovery, and process optimisation, where experiments are costly and evaluation budgets are limited.

Three lessons stand out:

- **Model confidence is not evidence       :** Function 8's relevance estimates and diagnostics consistently dismissed one input until a controlled test contradicted them at −4.50σ. Multiple agreeing diagnostics can share the same underlying blind spot.
- **Sampling shapes what you discover      :** Model-guided queries increasingly reflect existing assumptions. Clusters and feature importance can therefore reveal the sampling strategy rather than the true structure of the objective function.
- **Calibration matters more than fit alone:** Raising the noise floor worsened the in-sample fit but improved the honesty of predictions. Under a fixed budget, actionable uncertainty is more valuable than a fit that inspires unjustified confidence.

Ultimately, this project is not just about finding optima, it is about making decisions under uncertainty and documenting them so others can scrutinise the reasoning.

## Section 2: Inputs and Outputs

### Inputs

Each query is a vector of floats in `[0.000000, 0.999999]`, one per dimension, formatted to exactly six decimal places. Each query is submitted as a hyphen-separated string of values [x1-x2-x3-...-xn], where every value begins with 0 and is specified to six decimal places.

| | Description |
|---|---|
| Input | Initial observations per function (`data/initial/function<1-8>/initial_inputs.npy`) |
| Input | Submissions at the end of each week for all 8 functions (`data/accumulated/week<1-12>/inputs.txt`) |

### Outputs

Each query returns a single unbounded float.

| | Description |
|---|---|
| Output | Initial output per function (`data/initial/function<1-8>/initial_outputs.npy`) |
| Output | Scalar value returned after evaluating each query point for all 8 functions at the end of each week (`data/accumulated/week<1-12>/outputs.txt`) |

## Section 3: Technical Approach

All eight of the functions use the same setup: a Gaussian Process surrogate with an acquisition function driving each query. The reason is the situation rather than a preference for GPs. I get one expensive evaluation a week, no gradients, and no equation to inspect. My datasets range from about 15 points in two dimensions up to 45 points in eight. At that size, the thing that matters isn't how flexible a model is, it's whether it knows when it doesn't know. GPs give a prediction and an honest confidence interval together, and UCB and EI both need that second number to work at all. Srinivas et al. proved GP-UCB has no-regret guarantees under exactly this kind of budget, which is why I trust it over random or grid search.

**Papers that guided the design**

- *Gaussian Processes for Machine Learning - Rasmussen and Williams (2006)*: is what I keep going back to for kernel and length-scale reasoning. Its treatment of ARD length-scales is how I spotted that x8 in Function 8 carries no signal, since its length-scale ran straight to the bound.
- *Efficient Global Optimization of Expensive Black-Box Functions - Jones, Schonlau and Welch (1998)*: is where the whole fit-a-surrogate-then-pick-a-point loop comes from. It also explains why Expected Improvement collapses once the model believes nothing can beat the incumbent. I hit that exact failure, which told me EI was unusable there and UCB was the right call.
- *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design - Srinivas et al. (2010)*: gave me β as an explicit exploration knob rather than a number I'd guessed. It's why lowering β over time was principled. I dropped it faster than the theory suggests and then reversed course in Week 5.
- *Practical Bayesian Optimization of Machine Learning Algorithms - Snoek et al. (2012)*: shaped two things: that Matern 5/2 usually beats RBF on rough performance surfaces, which my cross-validation agreed with every week, and using ARD length-scales as a relevance measure. That second idea is what exposed the x3 problem in Function 7 and led to my best result.


## Section 4: Documentation

- [Model Card](docs/model_card.md)
- [Data Sheet](docs/datasheet.md)
- [Strategy](docs/strategy.md)
- [Methodology](docs/methodology.md)
- [Decision log](docs/decision_log.md)
- [Limitations](docs/limitations.md)
