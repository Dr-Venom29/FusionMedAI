# Research Document 02: TreeSHAP Algorithmic Formulation & Properties

## 1. Classical Shapley Values vs. TreeSHAP

In cooperative game theory, the Shapley value allocates payouts to players based on their marginal contributions across all possible coalitions $S \subseteq F \setminus \{j\}$:
$$\phi_j(v) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ v(S \cup \{j\}) - v(S) \right]$$

For general non-linear models with $D=119$ features, computing exact Shapley values requires evaluating $2^{119} \approx 6.6 \times 10^{35}$ coalitions, which is computationally intractable.

**TreeSHAP** (Lundberg et al., 2020) solves this in polynomial time $O(T \cdot L \cdot D^2)$ by recursively tracking conditional expectations down tree paths. For CatBoost's **symmetric oblivious decision trees**, where split predicates at depth $d$ are identical across all $2^d$ leaves, exact TreeSHAP is evaluated efficiently under the supported tree-model formulation.

---

## 2. Fundamental Axiomatic Guarantees

TreeSHAP satisfies four fundamental game-theoretic properties:

1. **Efficiency (Local Accuracy)**:
   The sum of feature attributions equals the difference between the model output $\hat{f}(x)$ and the baseline expected value $\mathbb{E}[\hat{f}(X)]$:
   $$\sum_{j=1}^{D} \phi_j(x) = \hat{f}(x) - \mathbb{E}[\hat{f}(X)]$$
   In our CatBoost model, the baseline expected value $\mathbb{E}[\hat{f}(X)] = -2.1686$ is obtained directly as the base value from the training data background distribution (the model's average tree output in margin log-odds space across the training partition, corresponding to base probability $\sigma(-2.1686) \approx 10.26\%$).

2. **Symmetry**:
   If two features $j$ and $k$ contribute equally to all possible feature subsets:
   $$v(S \cup \{j\}) = v(S \cup \{k\}) \quad \forall S \subseteq F \setminus \{j, k\} \implies \phi_j(x) = \phi_k(x)$$

3. **Dummy (Null Player)**:
   If feature $j$ never changes the prediction across any subtree:
   $$v(S \cup \{j\}) = v(S) \quad \forall S \subseteq F \setminus \{j\} \implies \phi_j(x) = 0$$

4. **Additivity (Consistency)**:
   For an ensemble of $T=350$ trees where $\hat{f}(x) = \sum_{t=1}^{T} f_t(x)$, the ensemble attribution is the direct sum of individual tree attributions:
   $$\phi_j(\hat{f}) = \sum_{t=1}^{T} \phi_j(f_t)$$

---

## 3. Log-Odds to Probability Transformation

TreeSHAP computes additive attributions in the model's unconstrained margin (log-odds) space:
$$\hat{z}(x) = \phi_0 + \sum_{j=1}^{D} \phi_j(x)$$
Where $\phi_0 = -2.1686$. The final predicted readmission probability $\hat{p}(x)$ is obtained via the standard logistic sigmoid:
$$\hat{p}(x) = \sigma\left(\hat{z}(x)\right) = \frac{1}{1 + \exp\left(-\left(\phi_0 + \sum_{j=1}^{D} \phi_j(x)\right)\right)}$$

This preserves exact interpretability: positive $\phi_j(x) > 0$ increases the log-odds (pushing $\hat{p}$ upward), while negative $\phi_j(x) < 0$ decreases the log-odds.
