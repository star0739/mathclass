# assessment/ai_loss.py
# ------------------------------------------------------------
# Loss function registry for AI math performance task
# - Supports 3 selectable loss surfaces with difficulty levels:
#   (1) Weighted quadratic (Lv1)
#   (5) Double well (Lv2)
#   (6) Banana valley / Rosenbrock-lite (Lv3)
#
# Design goals:
# - 1~2 parameters per loss type (student-friendly)
# - Stable defaults + parameter clamping (to reduce divergence)
# - Unified API: make_loss_spec / E / grad / latex_E / recommended_step_size
# - Works with scalars or numpy arrays (vectorized)
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple, Union

import numpy as np

LossType = Literal["quad", "double_well", "banana"]


# -----------------------------
# Metadata shown to students
# -----------------------------
LOSS_CATALOG: Dict[LossType, Dict[str, Any]] = {
    "quad": {
        "label": "Lv1 | 가중 이차형",
        "level": 1,
        "description": "축 정렬 타원 등고선. 계수에 따라 a/b 방향 민감도가 달라짐.",
        "params": ["alpha"],  # keep 1-parameter by default
        "latex_template": r"E(a,b)=\alpha a^{2}+b^{2}",
        "param_ranges": {
            "alpha": (2.0, 20.0),
        },
        "default_params": {"alpha": 10.0},
        "recommended_step": (0.12, 0.25),  # (min, max)
    },
    "double_well": {
        "label": "Lv2 | 이중 우물(다봉)",
        "level": 2,
        "description": "a축 방향으로 두 최소점(±1,0). 시작점에 따라 다른 최소점으로 수렴 가능.",
        "params": ["lam"],
        "latex_template": r"E(a,b)=(a^{2}-1)^{2}+\lambda b^{2}",
        "param_ranges": {
            "lam": (0.5, 6.0),
        },
        "default_params": {"lam": 2.0},
        "recommended_step": (0.03, 0.12),
    },
    "banana": {
        "label": "Lv3 | 바나나 골짜기",
        "level": 3,
        "description": "휘어진 골짜기(최소점 (1,1)). β가 클수록 골짜기가 좁고 수렴이 어려움.",
        "params": ["beta"],
        "latex_template": r"E(a,b)=(a-1)^{2}+\beta (b-a^{2})^{2}",
        "param_ranges": {
            "beta": (5.0, 40.0),
        },
        "default_params": {"beta": 15.0},
        "recommended_step": (0.002, 0.02),
    },
}


@dataclass(frozen=True)
class LossSpec:
    """Normalized loss spec used throughout step1/step2/report."""
    type: LossType
    params: Dict[str, float]

    # metadata (optional but handy)
    level: int
    label: str


Number = Union[float, int, np.ndarray]


# -----------------------------
# Helpers
# -----------------------------
def _to_array(x: Number) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _norm_spec(loss_type: LossType, params: Dict[str, Any] | None = None) -> LossSpec:
    if loss_type not in LOSS_CATALOG:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    meta = LOSS_CATALOG[loss_type]
    p = dict(meta["default_params"])  # start from defaults

    if params:
        for k, v in params.items():
            if k in p:
                try:
                    p[k] = float(v)
                except Exception:
                    # ignore invalid values; keep default
                    pass

    # clamp to safe ranges
    for k, (lo, hi) in meta["param_ranges"].items():
        p[k] = _clamp(p.get(k, lo), lo, hi)

    return LossSpec(
        type=loss_type,
        params=p,
        level=int(meta["level"]),
        label=str(meta["label"]),
    )


# -----------------------------
# Public API
# -----------------------------
def make_loss_spec(
    loss_type: LossType,
    params: Dict[str, Any] | None = None,
) -> LossSpec:
    """
    Create a normalized LossSpec with defaults + clamped parameters.
    Example:
        spec = make_loss_spec("quad", {"alpha": 12})
    """
    return _norm_spec(loss_type, params)


def loss_label(loss_type: LossType) -> str:
    return str(LOSS_CATALOG[loss_type]["label"])


def loss_level(loss_type: LossType) -> int:
    return int(LOSS_CATALOG[loss_type]["level"])


def param_ranges(loss_type: LossType) -> Dict[str, Tuple[float, float]]:
    """Ranges suitable for slider bounds."""
    return dict(LOSS_CATALOG[loss_type]["param_ranges"])


def default_params(loss_type: LossType) -> Dict[str, float]:
    return dict(LOSS_CATALOG[loss_type]["default_params"])


def recommended_step_size(loss_spec: LossSpec) -> float:
    """
    Provide a conservative recommended step size based on difficulty.
    Returns a single float (use as default in UI).
    """
    lo, hi = LOSS_CATALOG[loss_spec.type]["recommended_step"]
    # use lower-ish default for stability, but not too tiny
    return float((2 * lo + hi) / 3)


def recommended_step_range(loss_spec: LossSpec) -> Tuple[float, float]:
    """Return (min, max) recommended step size range for UI hints."""
    lo, hi = LOSS_CATALOG[loss_spec.type]["recommended_step"]
    return float(lo), float(hi)


def latex_E(loss_spec: LossSpec) -> str:
    """
    Return LaTeX-friendly expression of E(a,b) (without $...$).
    Uses ^{ } form for stability.
    """
    t = loss_spec.type
    p = loss_spec.params

    if t == "quad":
        a = p["alpha"]
        # E = alpha a^2 + b^2
        if abs(a - 1.0) < 1e-12:
            return r"E(a,b)=a^{2}+b^{2}"
        # show alpha compactly (avoid trailing .0)
        a_str = _fmt(a)
        return rf"E(a,b)={a_str}a^{{2}}+b^{{2}}"

    if t == "double_well":
        lam = p["lam"]
        lam_str = _fmt(lam)
        return rf"E(a,b)=(a^{{2}}-1)^{{2}}+{lam_str}b^{{2}}"

    if t == "banana":
        beta = p["beta"]
        beta_str = _fmt(beta)
        return rf"E(a,b)=(a-1)^{{2}}+{beta_str}(b-a^{{2}})^{{2}}"

    raise ValueError(f"Unknown loss type: {t}")


def E(a: Number, b: Number, loss_spec: LossSpec) -> np.ndarray:
    """
    Compute loss E(a,b) (vectorized).
    Returns numpy array (scalar if inputs are scalar).
    """
    t = loss_spec.type
    p = loss_spec.params
    a_ = _to_array(a)
    b_ = _to_array(b)

    if t == "quad":
        alpha = p["alpha"]
        return alpha * a_**2 + b_**2

    if t == "double_well":
        lam = p["lam"]
        return (a_**2 - 1.0) ** 2 + lam * (b_**2)

    if t == "banana":
        beta = p["beta"]
        return (a_ - 1.0) ** 2 + beta * (b_ - a_**2) ** 2

    raise ValueError(f"Unknown loss type: {t}")


def grad(a: Number, b: Number, loss_spec: LossSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient (dE/da, dE/db) (vectorized).
    Returns two numpy arrays (scalar if inputs are scalar).
    """
    t = loss_spec.type
    p = loss_spec.params
    a_ = _to_array(a)
    b_ = _to_array(b)

    if t == "quad":
        alpha = p["alpha"]
        dE_da = 2.0 * alpha * a_
        dE_db = 2.0 * b_
        return dE_da, dE_db

    if t == "double_well":
        lam = p["lam"]
        # d/da (a^2-1)^2 = 2(a^2-1)*2a = 4a(a^2-1)
        dE_da = 4.0 * a_ * (a_**2 - 1.0)
        dE_db = 2.0 * lam * b_
        return dE_da, dE_db

    if t == "banana":
        beta = p["beta"]
        # E = (a-1)^2 + beta (b-a^2)^2
        # dE/da = 2(a-1) + beta*2(b-a^2)*(-2a) = 2(a-1) - 4beta*a*(b-a^2)
        # dE/db = beta*2(b-a^2)
        dE_da = 2.0 * (a_ - 1.0) - 4.0 * beta * a_ * (b_ - a_**2)
        dE_db = 2.0 * beta * (b_ - a_**2)
        return dE_da, dE_db

    raise ValueError(f"Unknown loss type: {t}")


def clip_grad(
    dE_da: np.ndarray,
    dE_db: np.ndarray,
    max_norm: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional stabilizer: clip gradient vector norm to max_norm.
    Use in step2 if you want to reduce divergence risk.
    """
    g1 = _to_array(dE_da)
    g2 = _to_array(dE_db)
    norm = np.sqrt(g1**2 + g2**2)
    # avoid division by zero
    scale = np.where(norm > max_norm, max_norm / (norm + 1e-12), 1.0)
    return g1 * scale, g2 * scale


def safe_start_hint(loss_spec: LossSpec) -> Dict[str, Tuple[float, float]]:
    """
    Optional: recommended start ranges (for UI hints).
    Not enforced; meant as guidance to avoid extreme divergence.
    """
    t = loss_spec.type
    if t == "quad":
        return {"a": (-2.5, 2.5), "b": (-2.5, 2.5)}
    if t == "double_well":
        return {"a": (-2.5, 2.5), "b": (-2.0, 2.0)}
    if t == "banana":
        return {"a": (-2.0, 2.0), "b": (-1.0, 3.0)}
    return {"a": (-2.5, 2.5), "b": (-2.5, 2.5)}


# -----------------------------
# Formatting helpers
# -----------------------------
def _fmt(x: float) -> str:
    """
    Compact numeric formatter for LaTeX.
    Examples: 10.0 -> "10", 2.5 -> "2.5"
    """
    if abs(x - round(x)) < 1e-10:
        return str(int(round(x)))
    s = f"{x:.6g}"
    return s


# -----------------------------
# Convenience: catalog export
# -----------------------------
def catalog_rows() -> list[dict[str, Any]]:
    """
    Returns rows suitable for displaying in a table:
    [{'type':..., 'label':..., 'level':..., 'description':..., 'params':...}, ...]
    """
    rows: list[dict[str, Any]] = []
    for t, meta in LOSS_CATALOG.items():
        rows.append(
            {
                "type": t,
                "label": meta["label"],
                "level": meta["level"],
                "description": meta["description"],
                "params": ", ".join(meta["params"]),
                "latex": meta["latex_template"],
            }
        )
    return rows
