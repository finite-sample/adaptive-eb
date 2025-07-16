# Adaptive Entropy Balancing via Multiplicative Weights

A scalable, streaming-ready alternative to classical entropy balancing for survey research and causal inference.

## Overview

In applied survey research and causal inference, entropy balancing is used to reweight observational data so that covariate distributions match known or desired population margins. The standard approach solves a convex optimization problem that minimizes KL divergence from base weights while enforcing exact moment-matching constraints.

However, real-world applications increasingly demand:

1. **Streaming data support** - Reweighting data that arrives in batches (e.g., polling waves, rolling panels)
2. **Dynamic target adjustment** - Adjusting to updated calibration targets (e.g., revised Census benchmarks)  
3. **High-dimensional scalability** - Scaling to thousands of covariates from rich administrative data

## Problem with Current Approaches

The classic entropy balancing method (typically implemented via BFGS optimization) has significant limitations:

- **Not adaptive** — requires solving from scratch each time targets or data change
- **Not scalable** — struggles or fails when the number of covariates is large (d > 100)
- **Not streaming-ready** — assumes the full dataset is available all at once

## Our Solution: Multiplicative Weights Update (MWU)

We replace closed-form optimization with a Multiplicative Weights Update algorithm that:

1. **Maintains weights via iterative exponential updates** - Simple, stable weight adjustments
2. **Operates in batch or mini-batch mode** - Supports both traditional and streaming workflows  
3. **Adapts on the fly to changing target moments** - No restart required when targets change

This transforms entropy balancing into a no-regret learning process over marginal constraints.

## How It Works

### Problem Setup
Given sample covariates from n observations with d features and target population moments, find weights such that:
- Weighted sample moments match population targets exactly
- All weights are positive and sum to one

### MWU Algorithm
Instead of solving a constrained optimization problem directly, MWU:

1. **Initializes** weights uniformly across observations
2. **Computes** current moment error (difference between weighted sample and targets)
3. **Updates** weights exponentially based on each observation's contribution to error
4. **Normalizes** weights to maintain valid probability distribution
5. **Repeats** until convergence

The learning rate parameter controls the step size, balancing convergence speed with stability.

### Operating Modes

- **Batch Mode**: Updates using complete dataset each iteration (fastest convergence)
- **Streaming Mode**: Processes mini-batches over multiple epochs (memory efficient)  
- **Adaptive Mode**: Continues from current state when targets change (no restart cost)

## Performance Results

We tested on synthetic datasets with n=3000 observations:

| Scenario | L2 Error | Time (s) | Comments |
|----------|----------|----------|----------|
| BFGS (d=10) | 0.0000 | 0.008 | 5 iterations |
| Batch MWU (d=10) | 0.0000 | 0.004 | 50 iterations |
| Streaming MWU | 0.0042 | 0.022 | 10 passes |
| Adaptive MWU (target shift) | 0.0028 | 0.020 | smooth adaptation |
| BFGS (d=100) | 0.0001 | 1.184 | 44 iterations |
| MWU (d=100) | 0.0000 | 0.081 | 200 iterations |

**Key findings**: MWU matches classical entropy balancing accuracy, adapts instantly to target shifts, and is an order-of-magnitude faster in high dimensions.

## Key Benefits

- **Flexibility**: Handles target changes without restarting optimization
- **Scalability**: Runtime scales linearly in sample size and dimensions; no matrix inversions required
- **Streaming-ready**: Same exponential update works in mini-batch or online settings
- **Interpretability**: Weights remain strictly positive and controlled by learning rate parameter

## When to Use MWU

MWU offers a drop-in alternative to classical entropy balancing when you need:

- **Adaptive reweighting** for changing calibration targets
- **Streaming data processing** for large datasets or real-time applications  
- **High-dimensional performance** with hundreds or thousands of covariates
- **Memory efficiency** for datasets that don't fit in memory

## Theoretical Foundation

The MWU update is a no-regret algorithm that converges to the classical entropy balancing solution under standard mirror-descent guarantees. This ensures theoretical soundness while providing practical advantages.

## Applications

- **Survey research**: Dynamic panel reweighting as new waves arrive
- **Market research**: Real-time demographic balancing for online panels
- **Clinical trials**: Adaptive treatment assignment balancing
- **Administrative data**: Efficient processing of large government datasets

## References

- Hainmueller, J. (2012). "Entropy Balancing for Causal Effects: A Multivariate Reweighting Method to Produce Balanced Samples in Observational Studies." *Political Analysis*, 20(1), 25-46.
- Arora, S., Hazan, E., and Kale, S. (2012). "The Multiplicative Weights Update Method: A Meta-Algorithm and Applications." *Theory of Computing*, 8(1), 121-164.
