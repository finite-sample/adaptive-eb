## Adaptive Entropy Balancing via Multiplicative Weights

In applied survey research and causal inference, entropy balancing is used to reweight observational data so that covariate distributions match known or desired population margins. The standard approach solves a convex optimization problem that minimizes KL divergence from base weights while enforcing exact moment-matching constraints.

However, real-world applications increasingly demand:

1. Reweighting data that arrives in batches (e.g., polling waves, rolling panels),
2. Adjusting to updated calibration targets (e.g., revised Census benchmarks),
3. Scaling to high-dimensional covariates or large samples (thousands of covariates from rich administrative data)

### Limitations of Current Approaches

The classic entropy balancing method (e.g., via BFGS) is:

1. Not adaptive — it requires solving from scratch each time the targets or data change.
2. Not scalable — it can struggle or fail when the number of covariates is large (e.g., d = 100+).
3. Not streaming-ready — it assumes the full dataset is available all at once.

### MWU for Adaptive Weighting

We replace closed-form optimization with a Multiplicative Weights Update (MWU) algorithm, which:

1. Maintains weights via iterative exponential updates,
2. Can operate in batch or mini-batch (streaming) mode,
3. Adapts on the fly to changing target moments.

This turns entropy balancing into a no-regret learning process over marginal constraints.

### Problem Statement

Given sample covariates <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/42ee2a853a45ff2fe2ac506419128fe4.svg?invert_in_darkmode" align=middle width=148.56443744999999pt height=27.91243950000002pt/> and target population moments <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/5162ca930b220d2abb047d0d63f5c3bb.svg?invert_in_darkmode" align=middle width=70.45030574999998pt height=27.91243950000002pt/>, find weights <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/29546300c6306ca5a49fffe8206ad466.svg?invert_in_darkmode" align=middle width=73.19829494999999pt height=26.76175259999998pt/> such that

<p align="center"><img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/e00520177a006e8d517bda1726d3a022.svg?invert_in_darkmode" align=middle width=301.8876267pt height=44.89738935pt/></p>


Classical *entropy balancing* solves

<p align="center"><img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/11318ecb8ab80b19e82d7010805c8c28.svg?invert_in_darkmode" align=middle width=332.2149798pt height=23.835710249999998pt/></p>


where (<img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/6dbb78540bd76da3f1625782d42d6d16.svg?invert_in_darkmode" align=middle width=9.41027339999999pt height=14.15524440000002pt/>) are base weights (often uniform).

---

### Multiplicative‑Weights Reformulation

Writing the Lagrangian with dual <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/4b4199d8e43ea343cd11784015e33903.svg?invert_in_darkmode" align=middle width=48.395478449999985pt height=27.91243950000002pt/> gives    

<p align="center"><img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/aa83a1793d2f213e71d83229932c8e9e.svg?invert_in_darkmode" align=middle width=310.06587809999996pt height=49.38702615pt/></p>

A **mirror‑descent MWU** view keeps a weight vector <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/621d7edb8ca6049a16c1986d41805b9e.svg?invert_in_darkmode" align=middle width=27.45067049999999pt height=29.190975000000005pt/> and updates

<p align="center"><img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/8c3ff22ef92c59bfe816d0aa41a3ac8e.svg?invert_in_darkmode" align=middle width=296.4606843pt height=22.9224534pt/></p>

where <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/e804459fa3afd233b8a38063f2470bae.svg?invert_in_darkmode" align=middle width=201.25640369999996pt height=29.190975000000005pt/> is the current moment error and <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/3cb8a0a3372dc6708571e7b61e9fc9ec.svg?invert_in_darkmode" align=middle width=38.88877739999999pt height=21.18721440000001pt/> a learning rate.

* **Batch mode:** update using the full \$X\$.
* **Streaming mode:** apply the same exponential update to each mini‑batch; revisit weights over epochs.

The update (MWU) is a no‑regret algorithm that converges to the EB solution with rate <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/cf6d886f02baaf9e85874ef9080c9ca9.svg?invert_in_darkmode" align=middle width=68.35524629999999pt height=30.267491100000004pt/> under standard mirror‑descent guarantees.

---

### Simulation Example

* <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/df72ab238a07bc75109c4ff0f59667f3.svg?invert_in_darkmode" align=middle width=64.66134509999998pt height=21.18721440000001pt/>, <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/7d127c72445ef873b64eb74965c9f566.svg?invert_in_darkmode" align=middle width=46.91201294999998pt height=22.831056599999986pt/> covariates drawn i.i.d. <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/245ee30cc436b5977b4dc7bb5d765559.svg?invert_in_darkmode" align=middle width=52.73634794999999pt height=24.65753399999998pt/> and scaled.
* Population targets <img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/2a25eba856cde3ff95ae93c8b5177611.svg?invert_in_darkmode" align=middle width=42.253525049999986pt height=18.666631500000015pt/> sampled Uniform<img src="https://rawgit.com/finite-sample/adaptive-eb/main/svgs/0ae1d7c4775b0df0c9f4a3b79642e889.svg?invert_in_darkmode" align=middle width=74.8860354pt height=24.65753399999998pt/>.
* Algorithms compared:
  1. **BFGS** solution of (EB) (batch, closed‑form)
  2. **Batch MWU** (full data each iteration)
  3. **Streaming MWU** (mini‑batches of 50, 10 passes)
  4. **Adaptive MWU**: after 5 passes, targets shift and MWU continues.
* High‑dim test: \$d=100\$.

| Scenario             | L2 error | Time (s) | Comments          |
| -------------------- | -------: | -------: | ----------------- |
| BFGS (\$d=10\$)      |   0.0000 |    0.008 | 5 iters           |
| Batch MWU (\$d=10\$) |   0.0000 |    0.004 | 50 iters          |
| Streaming MWU        |   0.0042 |    0.022 | 10 passes         |
| Adaptive MWU (shift) |   0.0028 |    0.020 | smooth adaptation |
| BFGS (\$d=100\$)     |   0.0001 |    1.184 | 44 iters          |
| MWU (\$d=100\$)      |   0.0000 |    0.081 | 200 iters         |

*MWU matches EB accuracy, adapts instantly to target shifts, and is an order‑of‑magnitude faster in high dimensions.*

---

### Key Takeaways

* **Flexibility:** MWU handles target changes without restarting optimization.
* **Scalability:** Runtime scales linearly in \$n,d\$; no matrix inversions.
* **Streaming‑ready:** Same exponential update works in mini‑batch or online settings.
* **Interpretability:** Weights remain strictly positive and controlled by \$\eta\$.

Hence, MWU offers a drop‑in alternative to classical entropy balancing when adaptivity, streaming data, or high‑dimensional covariates matter.

---

### References

* Hainmueller, J. (2012). "Entropy Balancing …" *Political Analysis*.
* Arora, Hazan & Kale (2012). "The Multiplicative Weights Update Method: a Meta‑Algorithm …"
