"""Convergence CSV utilities.

The solver can emit two kinds of convergence CSVs.

Time convergence (from `Wave::write_time_convergence_csv`)
  columns: dt,u_L2,u_H1,v_L2,q_uL2,q_uH1,q_vL2

Space convergence (from `Wave::write_space_convergence_csv`)
  columns: h,u_L2,u_H1,v_L2,p_uL2,p_uH1,p_vL2,mesh

This module provides:
- robust loading/coercion
- detection of CSV kind
- observed-order estimation (per-step and global fit in log-log space)

Design goals:
- no extra dependencies beyond numpy/pandas
- tolerate "nan" strings and non-numeric mesh paths
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class ConvergenceKind(str, Enum):
    SPACE = "space"
    TIME = "time"


@dataclass(frozen=True)
class FitResult:
    slope: float
    intercept: float
    r2: float
    n: int


_SPACE_REQUIRED = {"h", "u_L2", "u_H1", "v_L2"}
_TIME_REQUIRED = {"dt", "u_L2", "u_H1", "v_L2"}


def detect_kind(df: pd.DataFrame) -> ConvergenceKind:
    cols = set(df.columns)
    if "h" in cols:
        return ConvergenceKind.SPACE
    if "dt" in cols:
        return ConvergenceKind.TIME
    raise ValueError("Could not detect convergence kind: missing 'h' or 'dt' column")


def load_convergence_csv(file_like, *, name: str | None = None) -> pd.DataFrame:
    """Load a convergence CSV, coercing numeric columns.

    `file_like` can be a file path or a file-like object.
    """
    df = pd.read_csv(file_like)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    numeric_cols = [
        "h",
        "dt",
        "u_L2",
        "u_H1",
        "v_L2",
        "p_uL2",
        "p_uH1",
        "p_vL2",
        "q_uL2",
        "q_uH1",
        "q_vL2",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "mesh" in df.columns:
        # keep mesh paths intact (and avoid pandas treating them as NaN)
        df["mesh"] = df["mesh"].astype(str)

    if name is not None and "__name" not in df.columns:
        df["__name"] = name

    kind = detect_kind(df)
    req = _SPACE_REQUIRED if kind == ConvergenceKind.SPACE else _TIME_REQUIRED
    missing = sorted(req.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for {kind.value} convergence CSV: {missing}")

    return df


def _safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(x > 0, x, np.nan)
    return np.log10(x)


def fit_order(x: Iterable[float], y: Iterable[float]) -> FitResult:
    """Fit y ≈ C * x^p in log10 space.

    Returns slope p and intercept log10(C). Non-positive/NaN values are ignored.
    """
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)

    lx = _safe_log10(x)
    ly = _safe_log10(y)

    mask = np.isfinite(lx) & np.isfinite(ly)
    lx = lx[mask]
    ly = ly[mask]
    n = int(lx.size)

    if n < 2:
        return FitResult(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=n)

    slope, intercept = np.polyfit(lx, ly, 1)

    yhat = slope * lx + intercept
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - float(np.mean(ly))) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

    return FitResult(slope=float(slope), intercept=float(intercept), r2=r2, n=n)


def compute_observed_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row observed orders between consecutive rows.

    Adds/overwrites:
      - p_uL2, p_uH1, p_vL2 (space)
      - q_uL2, q_uH1, q_vL2 (time)

    Uses same formula as the C++ helper:
        log(e2/e1) / log(x2/x1)

    Notes:
      - Assumes the input rows are already ordered (coarse -> fine) as you want.
      - Works even if x decreases (e.g. dt: 0.1 -> 0.05).
    """
    out = df.copy()
    kind = detect_kind(out)

    xcol = "h" if kind == ConvergenceKind.SPACE else "dt"
    prefix = "p_" if kind == ConvergenceKind.SPACE else "q_"

    x = out[xcol].to_numpy(dtype=float)

    def _order(err_col: str) -> np.ndarray:
        e = out[err_col].to_numpy(dtype=float)
        res = np.full_like(e, np.nan, dtype=float)
        for i in range(1, len(e)):
            e1, e2 = e[i - 1], e[i]
            x1, x2 = x[i - 1], x[i]
            if not (np.isfinite(e1) and np.isfinite(e2) and np.isfinite(x1) and np.isfinite(x2)):
                continue
            if e1 <= 0 or e2 <= 0 or x1 <= 0 or x2 <= 0 or x1 == x2:
                continue
            res[i] = float(np.log(e2 / e1) / np.log(x2 / x1))
        return res

    out[f"{prefix}uL2"] = _order("u_L2")
    out[f"{prefix}uH1"] = _order("u_H1")
    out[f"{prefix}vL2"] = _order("v_L2")

    return out


def default_error_columns() -> list[str]:
    return ["u_L2", "u_H1", "v_L2"]


def pretty_metric_name(c: str) -> str:
    return {
        "u_L2": "u (L2)",
        "u_H1": "u (H1)",
        "v_L2": "v (L2)",
    }.get(c, c)


def summarize_fits(df: pd.DataFrame, *, error_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Create a small summary table with fitted slopes for each error column."""
    kind = detect_kind(df)
    xcol = "h" if kind == ConvergenceKind.SPACE else "dt"

    if error_cols is None:
        error_cols = default_error_columns()

    rows = []
    for ec in error_cols:
        if ec not in df.columns:
            continue
        fit = fit_order(df[xcol], df[ec])
        rows.append({"metric": ec, "slope": fit.slope, "r2": fit.r2, "n": fit.n})

    return pd.DataFrame(rows)
