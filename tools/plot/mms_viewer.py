# Refactored Wave CSV Viewer
# Single upload -> auto-detect -> auto-plot
#
# This Streamlit app:
# - accepts arbitrary CSV files
# - detects whether they match Wave solver output schemas
# - classifies them (MMS / Study / Convergence)
# - renders the appropriate plots & tables automatically

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Imports compatible with BOTH repo execution and standalone execution
# -----------------------------------------------------------------------------
try:
    # Repo layout (tools/plot/)
    from tools.plot.csv_utils import read_csv_robust
    from tools.plot.plot_convergence import build_figure as build_convergence_figure
    from tools.plot.convergence_utils import (
        compute_observed_orders,
        summarize_fits,
    )
except Exception:
    # Standalone layout (same folder)
    from csv_utils import read_csv_robust
    from plot_convergence import build_figure as build_convergence_figure
    from convergence_utils import (
        compute_observed_orders,
        summarize_fits,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


@dataclass
class ClassificationResult:
    recognized: bool
    kind: Literal[
        "mms",
        "study_dissipation",
        "study_modal",
        "conv",
        "unknown",
    ]
    confidence: float
    reasons: List[str]
    x_candidates: List[str]


def classify_csv(df: pd.DataFrame, filename: str) -> ClassificationResult:
    cols = set(df.columns)
    name = (filename or "").lower()

    def has_all(req: set[str]) -> bool:
        return all(c in cols for c in req)

    def has_any(opts: set[str]) -> bool:
        return any(c in cols for c in opts)

    # -------------------------------------------------------------------------
    # Study – dissipation
    # n, t, e, (optional e_over_e0)
    # -------------------------------------------------------------------------
    if has_all({"n", "t", "e"}):
        score = 3.0
        if "e_over_e0" in cols:
            score += 1.0
        if "dissip" in name:
            score += 0.2
        return ClassificationResult(
            True, "study_dissipation", score, ["found n,t,e"], ["t", "n"]
        )

    # -------------------------------------------------------------------------
    # Study – modal
    # n, t, a, adot
    # -------------------------------------------------------------------------
    if has_all({"n", "t", "a", "adot"}):
        score = 4.0 + (0.2 if "modal" in name else 0.0)
        return ClassificationResult(
            True, "study_modal", score, ["found n,t,a,adot"], ["t", "n"]
        )

    # -------------------------------------------------------------------------
    # Convergence (space or time)
    # h or dt + at least one error norm
    # -------------------------------------------------------------------------
    if ("h" in cols or "dt" in cols) and has_any({"u_l2", "u_h1", "v_l2"}):
        score = 4.0 + (0.2 if "conv" in name else 0.0)
        x_candidates = ["dt"] if "dt" in cols else ["h"]
        return ClassificationResult(
            True, "conv", score, ["found convergence schema"], x_candidates
        )

    # -------------------------------------------------------------------------
    # MMS
    # error_* metrics + time-like axis
    # -------------------------------------------------------------------------
    if any(c.startswith("error_") for c in cols) and has_any({"time", "t", "step", "n"}):
        score = 4.0 + (0.2 if "mms" in name else 0.0)
        x_candidates = [c for c in ["time", "t", "step", "n"] if c in cols]
        return ClassificationResult(
            True, "mms", score, ["found MMS error metrics"], x_candidates
        )

    return ClassificationResult(False, "unknown", 0.0, ["unrecognized schema"], [])


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Wave CSV Inspector", layout="wide")
st.title("Wave Solver CSV Inspector & Viewer")

uploaded_files = st.file_uploader(
    "Upload CSV files (analysis type will be auto-detected).",
    type=["csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more CSV files to begin.")
    st.stop()

for uf in uploaded_files:
    st.markdown("---")
    st.subheader(uf.name)

    try:
        df = read_csv_robust(uf.read())
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        continue

    df = normalize_columns(df)
    res = classify_csv(df, uf.name)

    if not res.recognized:
        st.error("❌ Unknown CSV format (not recognized as Wave solver output)")
        st.write("Columns found:", sorted(df.columns))
        st.dataframe(df.head(50))
        continue

    KIND_LABELS = {
        "mms": "Manufactured Solution (MMS) Analysis",
        "study_dissipation": "Energy Dissipation Study",
        "study_modal": "Modal Analysis",
        "conv": "Convergence Analysis",
    }

    st.markdown(
        f"""
    ### {KIND_LABELS.get(res.kind, res.kind)}
    *Automatically classified from CSV structure*
    """
    )

    # -------------------------------------------------------------------------
    # Dispatch plotting
    # -------------------------------------------------------------------------
    if res.kind == "study_dissipation":
        st.write("### Energy vs time")
        st.line_chart(df.set_index("t")[["e"]])

        if "e_over_e0" in df.columns:
            st.write("### E / E₀ vs time")
            st.line_chart(df.set_index("t")[["e_over_e0"]])

    elif res.kind == "study_modal":
        st.write("### Modal amplitude and velocity")
        st.line_chart(df.set_index("t")[["a", "adot"]])

    elif res.kind == "conv":
        # Reconstruct canonical convergence column names
        df_conv = df.copy()
        rename_map = {
            "u_l2": "u_L2",
            "u_h1": "u_H1",
            "v_l2": "v_L2",
            "p_ul2": "p_uL2",
            "p_uh1": "p_uH1",
            "p_vl2": "p_vL2",
            "q_ul2": "q_uL2",
            "q_uh1": "q_uH1",
            "q_vl2": "q_vL2",
        }
        df_conv = df_conv.rename(columns={k: v for k, v in rename_map.items() if k in df_conv.columns})

        st.write("### Convergence (log–log)")
        try:
            fig = build_convergence_figure(df_conv)
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Failed to build convergence plot: {e}")

        try:
            st.write("### Fitted slopes (observed order)")
            st.dataframe(summarize_fits(df_conv))
        except Exception as e:
            st.warning(f"Could not compute fitted slopes: {e}")

        try:
            st.write("### Per-step observed orders")
            st.dataframe(compute_observed_orders(df_conv))
        except Exception as e:
            st.warning(f"Could not compute per-step observed orders: {e}")

    elif res.kind == "mms":
        metrics = [c for c in df.columns if c.startswith("error_")]
        x = "time" if "time" in df.columns else ("t" if "t" in df.columns else res.x_candidates[0])
        st.write(f"### MMS error history vs `{x}`")
        st.line_chart(df.set_index(x)[metrics])

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')
