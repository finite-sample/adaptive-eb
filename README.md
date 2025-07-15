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

Given sample covariates \(x_i \in \mathbb{R}^d,\; i = 1,\dots,n\) and target population moments \(\bar{x}_{\text{pop}} \in \mathbb{R}^d\), find weights \(w \in \Delta^{\,n-1}\) such that

$$\[
  \sum_{i=1}^n w_i\,x_i \;=\; \bar{x}_{\text{pop}},
  \qquad
  w_i > 0,\;
  \sum_{i=1}^n w_i = 1.
\]$$

Classical *entropy balancing* solves

$$\[
  \min_{w}\;\mathrm{KL}\bigl(w \;\|\; u\bigr)
  \quad\text{s.t. the moment constraints},
  \tag{EB}
\]$$

where \(u\) are base weights (often uniform).

---

### Multiplicative‑Weights Reformulation

Writing the Lagrangian with dual \$\lambda\in\mathbb R^d\$ gives
$$
w\_i(\lambda)=\frac{u_i\,e^{-\lambda^\top x_i}}{Z(\lambda)},\quad Z(\lambda)=\sum_j u_j e^{-\lambda^\top x_j}
$$

A **mirror‑descent / MWU** view keeps a weight vector \$w^{(t)}\$ and updates

$$
w^{(t+1)}_i;\propto;w^{(t)}_i,\exp\bigl(-\eta, (x_i-\bar x_\text{pop})^\top g^{(t)}\bigr),\tag{MWU}
$$

where \$g^{(t)}=\sum\_i w^{(t)}\_i x\_i-\bar x\_\text{pop}\$ is the current moment error and \$\eta>0\$ a learning rate.

* **Batch mode:** update using the full \$X\$.
* **Streaming mode:** apply the same exponential update to each mini‑batch; revisit weights over epochs.

The update (MWU) is a no‑regret algorithm that converges to the EB solution with rate \$\tilde{\mathcal O}(1/\sqrt{T})\$ under standard mirror‑descent guarantees.

---

### Simulation Example

* \$n=3000\$, \$d=10\$ covariates drawn i.i.d. \$\mathcal N(0,I)\$ and scaled.
* Population targets \$\bar x\_{\text{pop}}\$ sampled Uniform$\[-0.3,0.3]\$.
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
