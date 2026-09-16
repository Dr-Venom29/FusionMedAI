# Chapter 03 — MC Dropout Pass Count Convergence

## 1. Experimental Protocol

To determine the optimal stochastic pass count $N^{*}$, a pass count convergence study was conducted on the validation split ($N=1,006$ images) across $N \in \{5, 10, 15, 20, 25, 30\}$.

Primary convergence selection evaluates Predictive Entropy stabilization ($\Delta H \le 10^{-3}$ nats) and Predictive Variance stabilization ($\Delta \text{Var} \le 10^{-4}$) for all subsequent increments.

---

## 2. Empirical Convergence Results

| Pass Count ($N$) | Mean Predictive Entropy (nats) | Mean Predictive Variance | Delta Entropy ($\Delta H$) | Delta Variance ($\Delta \text{Var}$) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 0.8172 | 0.000429 | — | — | Initial |
| **10** | **0.8175** | **0.000428** | **0.000315** | **0.000001** | **CONVERGED ($N^*$)** |
| 15 | 0.8174 | 0.000425 | 0.000130 | 0.000003 | Stable |
| 20 | 0.8174 | 0.000426 | 0.000035 | 0.000001 | Stable |
| 25 | 0.8175 | 0.000427 | 0.000074 | 0.000001 | Stable |
| 30 | 0.8174 | 0.000425 | 0.000082 | 0.000002 | Asymptotic |

---

## 3. Empirical Selection & Freeze

- **Convergence Criterion**: $\Delta H \le 10^{-3}$ nats AND $\Delta \text{Var} \le 10^{-4}$.
- At $N=10$, the change in mean predictive entropy is $0.000315$ nats ($\le 10^{-3}$) and mean predictive variance change is $0.000001$ ($\le 10^{-4}$), with all subsequent pass count increments remaining strictly within stabilization thresholds.
- **Empirically Selected Parameter**: $N^{*} = 10$ stochastic passes dynamically selected and frozen for held-out test evaluations.
