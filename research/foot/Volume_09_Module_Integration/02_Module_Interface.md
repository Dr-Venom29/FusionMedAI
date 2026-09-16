# Chapter 02 — Module Interface Contract & Input Validation

## 1. Class Definition & Initialization

The primary entry point for the foot modality pipeline is `FootModule` defined in `src/foot/foot_module.py`:

```python
class FootModule:
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        uncertainty_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> None:
        ...
```

Upon initialization, `FootModule`:
1. Resolves canonical filesystem paths for `best_model.pt`, `calibration.json`, and `uncertainty.json`.
2. Instantiates `FootEfficientNetB3` and loads frozen checkpoint weights.
3. Loads frozen Vector Scaling parameters ($w^{*}, b^{*}$) into `FootVectorScaler`.
4. Reads frozen uncertainty configuration ($N^{*}=10$).
5. Prepares `get_foot_val_transforms(image_size=224)` preprocessing pipeline.
6. Instantiates `FootGradCAM` linked to target layer `backbone.features.8`.

---

## 2. Primary Method Signature

```python
def predict(
    self,
    image: Union[str, Path, Image.Image],
    mc_passes: Optional[int] = None,
    generate_cam: bool = True
) -> Dict[str, Any]:
    ...
```

### Parameters:
- `image` (`Union[str, Path, Image.Image]`): Path string, `Path` object, or pre-loaded PIL `Image.Image`.
- `mc_passes` (`Optional[int]`): Number of Monte Carlo stochastic passes. If `None`, defaults to frozen $N^{*}=10$. If `0`, disables MC passes and returns deterministic metrics.
- `generate_cam` (`bool`): If `True`, computes Grad-CAM overlay and heatmap arrays. If `False`, skips CAM computation for accelerated inference.

---

## 3. Input Validation Protocol

Before model forward pass, inputs undergo strict boundary verification:

```
Input Argument (str / Path / PIL.Image)
           │
           ▼
     File Exists? ──────────► No  ──► Raise FileNotFoundError
           │
          Yes
           │
     Readable File? ────────► No  ──► Raise ValueError
           │
          Yes
           │
    PIL Decode & RGB ───────► Fail ─► Raise ValueError
           │
          Pass
           │
    Dimensions >= 10x10? ───► No  ──► Raise ValueError
           │
          Pass
           │
   Resize & Transform (224x224 RGB Tensor)
```

### Safety Distinction:
Input validation failures (missing file, non-image format, corrupted bytes) raise explicit standard Python exceptions at the boundary, ensuring input errors are clearly distinguished from valid model predictions and high prediction uncertainty states.
