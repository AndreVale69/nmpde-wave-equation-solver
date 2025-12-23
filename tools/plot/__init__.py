"""Plotting helpers package.

This package is used by Streamlit/CLI utilities under tools/plot.
"""

try:
    from .convergence_utils import (
        ConvergenceKind,
        FitResult,
        compute_observed_orders,
        detect_kind,
        fit_order,
        load_convergence_csv,
        pretty_metric_name,
        summarize_fits,
    )
except Exception:  # pragma: no cover
    # In some IDE contexts, the package root isn't resolved correctly.
    from convergence_utils import (  # type: ignore
        ConvergenceKind,
        FitResult,
        compute_observed_orders,
        detect_kind,
        fit_order,
        load_convergence_csv,
        pretty_metric_name,
        summarize_fits,
    )

__all__ = [
    "ConvergenceKind",
    "FitResult",
    "compute_observed_orders",
    "detect_kind",
    "fit_order",
    "load_convergence_csv",
    "pretty_metric_name",
    "summarize_fits",
]
