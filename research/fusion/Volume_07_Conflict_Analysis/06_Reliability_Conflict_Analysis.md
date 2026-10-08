# Global Reliability & Authority Alignment Analysis

## 1. Frozen Global Reliability Hierarchy

The pre-registered validation reliability priors are:

$$R_R = 0.929956 \quad (\text{Retina})$$

$$R_F = 0.922266 \quad (\text{Foot})$$

$$R_C = 0.825382 \quad (\text{Clinical})$$

Ordering: $R_R > R_F > R_C$.

---

## 2. Pairwise Authority Dominance Under Conflict

When conflicting modalities project divergent risks, ACARA-U dynamic routing weights reflect both instance confidence/quality and global empirical validation priors:

| Modality Pair Comparison | Higher-Reliability Modality | Lower-Reliability Modality | Authority Dominance Rate ($w_{\text{high}} > w_{\text{low}}$) |
| :--- | :--- | :--- | :---: |
| **Retina vs Clinical** | Retina ($R_R = 0.929956$) | Clinical ($R_C = 0.825382$) | **98.0%** (490 / 500) |
| **Retina vs Foot** | Retina ($R_R = 0.929956$) | Foot ($R_F = 0.922266$) | **93.2%** (466 / 500) |
| **Foot vs Clinical** | Foot ($R_F = 0.922266$) | Clinical ($R_C = 0.825382$) | **50.2%** (251 / 500) |

---

## 3. Scientific Observations

1. **Retina Dominance Consistency**: Due to higher global reliability ($R_R = 0.930$) and low mean uncertainty ($U_R = 0.082$), Retina is assigned dominant authority in $>93\%$ of conflicting pairwise encounters.
2. **Foot vs Clinical Dynamic Trade-Off**: Foot possesses a higher reliability prior ($0.922$ vs $0.825$), but higher mean uncertainty ($\overline{U_F} = 0.528$ vs $\overline{U_C} = 0.024$). Consequently, the router balances prior reliability against instance uncertainty, allocating greater weight to Foot in $50.2\%$ and to Clinical in $49.8\%$ of packets.
