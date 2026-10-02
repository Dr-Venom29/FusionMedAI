# 03 ACARA-U Router Input & Output Contracts — Phase C11.4

## 1. Input Contract Specifications

### 1.1 `ModalityChannelInput`
Immutable container encapsulating one modality's routing attributes:

```python
@dataclass(frozen=True)
class ModalityChannelInput:
    modality: str        # 'retina', 'foot', or 'clinical'
    confidence: float    # C_i in [0.0, 1.0]
    reliability: float   # R_i in [0.0, 1.0] (must match frozen prior)
    uncertainty: float   # U_i in [0.0, 1.0], common directional routing scale
    quality: float       # Q_i in [0.0, 1.0]
    availability: bool   # A_i in {True, False}
```

### Invariant Checks:
1. **Scalar Boundedness**: $C_i, R_i, U_i, Q_i \in [0.0, 1.0]$ and non-NaN / finite.
2. **Hard Availability Quality Rule**: $A_i = \text{False} \implies Q_i = 0.0$.
3. **Reliability Prior Match**: $|R_i - R_i^{\text{frozen}}| \le 10^{-5}$.

---

### 1.2 `RouterInput`
Container bundling the three modality channels:

```python
@dataclass(frozen=True)
class RouterInput:
    retina: ModalityChannelInput
    foot: ModalityChannelInput
    clinical: ModalityChannelInput
```

---

## 2. Output Contract Specification: `RouterResult`

```python
@dataclass(frozen=True)
class RouterResult:
    weights: Dict[str, float]           # {'retina': w_R, 'foot': w_F, 'clinical': w_C}
    logits: Dict[str, float]            # Raw pre-masking logits z_i
    masked_logits: Dict[str, float]     # Post-masking logits z_tilde_i
    availability: Dict[str, bool]       # Modality availability flags
    coefficients: Dict[str, float]      # Applied hyperparameters Theta
    active_modalities: List[str]        # Available modality identifiers
    num_active: int                     # |A| in 0..3
    normalization_sum: float            # Strictly 1.0 (or 0.0 if empty)
    routing_entropy: float              # H(w) in [0.0, ln(3)]
    dominant_modality: Optional[str]    # Modality with max weight
    status: str                         # 'SUCCESS' or 'NO_MODALITY_AVAILABLE'
    router_version: str                 # 'acarau_v2.0'
```

---

## 3. Scope Note

`RouterResult.weights` represent **decision authority allocation**, not probabilities of a common clinical outcome. The router does not produce $R_{\text{fusion}}$, DCRI, or a clinically calibrated multimodal probability.
