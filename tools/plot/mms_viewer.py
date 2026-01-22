# Refactored Wave CSV Viewer
# Single upload -> auto-detect -> auto-plot
#
# This Streamlit app:
# - accepts arbitrary CSV files
# - detects whether they match Wave solver output schemas
# - classifies them (MMS / Study / Convergence)
# - renders the appropriate plots & tables automatically

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
import streamlit as st

try:
    from tools.plot.utils.generate_markdown import ollama_generate_markdown, ollama_generate_markdown_stream
    from tools.plot.utils.llm_utils import build_interpretation_prompt
    from tools.plot.utils.csv_utils import read_csv_robust
    from tools.plot.plot_convergence import build_figure as build_convergence_figure
    from tools.plot.utils.convergence_utils import (
        compute_observed_orders,
        summarize_fits,
    )
    from tools.plot.markdown.modal_explanation import modal_explanation_for_dummies
    from tools.plot.markdown.modal_explanation import modal_explanation_text
    from tools.plot.markdown.dissipation_explanation import dissipation_explanation, dissipation_explanation_for_dummies
    from tools.plot.markdown.convergence_explanation import convergence_explanation, convergence_explanation_for_dummies
    from tools.plot.markdown.mms_explanation import mms_explanation, mms_explanation_for_dummies
    from tools.plot.utils.plot_comparison import plot_compare_timeseries, compare_expander_timeseries
except Exception:
    # Standalone layout (same folder)
    from utils.generate_markdown import ollama_generate_markdown, ollama_generate_markdown_stream
    from utils.llm_utils import build_interpretation_prompt
    from utils.csv_utils import read_csv_robust
    from plot_convergence import build_figure as build_convergence_figure
    from utils.convergence_utils import (
        compute_observed_orders,
        summarize_fits,
    )
    from markdown.modal_explanation import modal_explanation_for_dummies
    from markdown.modal_explanation import modal_explanation_text
    from markdown.dissipation_explanation import dissipation_explanation, dissipation_explanation_for_dummies
    from markdown.convergence_explanation import convergence_explanation, convergence_explanation_for_dummies
    from markdown.mms_explanation import mms_explanation, mms_explanation_for_dummies
    from utils.plot_comparison import plot_compare_timeseries, compare_expander_timeseries

if "parsed_by_kind" not in st.session_state:
    st.session_state["parsed_by_kind"] = {}


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


def ai_expander(
        prefix: str, _res: ClassificationResult, _uploaded_file
) -> None:
    with st.expander(f"🤖 AI interpretation ({prefix})"):
        api_key = None
        with st.expander("🔑 Setting up Ollama API access"):
            st.markdown(r"""
To use this feature, you need access to an Ollama-compatible model (hosted or local).
If you want to run a cloud-hosted model, you must have an Ollama account and API key.

### Steps to get an Ollama API key and set it up (safest method)

1. Sign up at https://ollama.com/
2. Once logged in, go to Settings: https://ollama.com/settings
3. Generate an API key from https://ollama.com/settings/keys and click "Add API Key".
4. Set a useful name and copy the generated key.
5. The API key must be set:
    - If you are running this app locally, set the environment variable `OLLAMA_API_KEY` before starting Streamlit.
    - If you are using Docker, set the environment variable in the Docker run command:
      ```bash
      docker build -t nm4pde-mms-viewer tools/plot
      docker run --rm -p 8501:8501 -e OLLAMA_API_KEY="your_api_key_here" nm4pde-mms-viewer
      ```
6. Restart the app if it was already running.

---

### Manually specifying API key (not recommended)

This is not recommended for security reasons, but if you want to specify the API key directly in the app,
you can do so by setting the `OLLAMA_API_KEY` environment variable in the app code before making any requests.
""")
            # create input box for API key (optional)
            api_key = st.text_input("Ollama API Key (optional)",
                                    type="password",
                                    key="ollama_api_key_" + _uploaded_file.name,
                                    placeholder="Enter your API key here (or leave blank to use env var)")

        # if API key provided, set env var
        if api_key:
            os.environ["OLLAMA_API_KEY"] = api_key

        host = st.text_input("Ollama host", value="https://ollama.com", key="ollama_host_" + _uploaded_file.name,
                             help="Ollama host URL (default: https://ollama.com for cloud-hosted models). For local Ollama server, use http://localhost:11434")
        model = st.text_input("Model", value="gpt-oss:120b-cloud", key="ollama_model_" + _uploaded_file.name,
                              help="Ollama-compatible model name (e.g., gpt-oss:120b-cloud). See https://ollama.com/search for options.")

        st.info(
            "For best results, use a cloud-hosted model (e.g., `gpt-oss:120b-cloud`) to avoid local resource issues.",
            icon="💡")

        if st.button(
                "Generate interpretation",
                key=f"ollama_btn_{_uploaded_file.name}",
                type="secondary",
        ):
            try:
                with st.spinner("🤖 Generating interpretation...", show_time=True):
                    prompt = build_interpretation_prompt(_res.kind, df)

                    out = st.empty()
                    acc = ""

                    for chunk in ollama_generate_markdown_stream(
                            prompt=prompt,
                            model=model,
                            host=host,
                    ):
                        acc += chunk
                        out.markdown(acc)

                out.markdown(acc)

            except Exception as e:
                st.error(f"Ollama error: {e}")


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

st.session_state["parsed_by_kind"] = {}
parsed_items = []

# -------------------------
# PASS 1: parse + classify + register
# -------------------------
for uf in uploaded_files:
    try:
        df = read_csv_robust(uf.read())
        df = normalize_columns(df)
        res = classify_csv(df, uf.name)
        error = None
    except Exception as e:
        df = None
        res = None
        error = f"Failed to parse CSV: {e}"

    parsed_items.append({
        "uf": uf,
        "df": df,
        "res": res,
        "error": error
    })

    if error is None and res.recognized:
        st.session_state["parsed_by_kind"].setdefault(res.kind, [])
        st.session_state["parsed_by_kind"][res.kind].append({"name": uf.name, "df": df.copy()})

for item in parsed_items:
    uf = item["uf"]
    df = item["df"]
    res = item["res"]

    st.markdown("---")
    st.subheader(uf.name)

    if item["error"] is not None:
        st.error(item["error"])
        continue

    if not res.recognized:
        st.error("❌ Unknown CSV format (not recognized as Wave solver output)")
        st.write("Columns found:", sorted(df.columns))
        st.dataframe(df.head(50))
        continue

    # -------------------------------------------------------------------------
    # Dispatch plotting
    # -------------------------------------------------------------------------
    if res.kind == "study_dissipation":
        if "E" in df.columns:
            st.write("#### Energy summary")

            E0 = df["E"].iloc[0]
            E_end = df["E"].iloc[-1]
            rel_change = (E_end - E0) / E0 if E0 != 0 else float("nan")

            summary = pd.DataFrame(
                {
                    "Initial energy E(0)": [E0],
                    "Final energy E(T)": [E_end],
                    "Relative change (E(T)-E(0))/E(0)": [rel_change],
                }
            )

            st.dataframe(summary, width='stretch')

        if "E_over_E0" in df.columns:
            st.write("#### Normalized energy extrema")
            extrema = pd.DataFrame(
                {
                    "min(E/E0)": [df["E_over_E0"].min()],
                    "max(E/E0)": [df["E_over_E0"].max()],
                }
            )
            st.dataframe(extrema, width='stretch')

        st.write("### Energy vs time")
        # st.line_chart(df.set_index("t")[["e"]])
        fig = plot_compare_timeseries({uf.name: df}, "t", ["e"], title="Energy vs time")
        st.plotly_chart(fig, width="stretch")

        if "e_over_e0" in df.columns:
            st.write("### $E / E_0$ vs time")
            fig = plot_compare_timeseries({uf.name: df}, "t", ["e_over_e0"], title="Normalized Energy E / E₀ vs time")
            st.plotly_chart(fig, width="stretch")

        y_cols = []
        if "e" in df.columns:
            y_cols.append("e")
        if "e_over_e0" in df.columns:
            y_cols.append("e_over_e0")
        # Default: plot current file alone (Plotly = legend toggling)
        if y_cols:
            st.write("### Energy time series")
            fig = plot_compare_timeseries({uf.name: df}, "t", y_cols, title="Energy time series")
            st.plotly_chart(fig, width="stretch")

        compare_expander_timeseries(
            kind="study_dissipation",
            current_name=uf.name,
            current_df=df,
            x_col="t",
            y_cols=y_cols,
            title="Energy comparison",
            expander_label="🔁 Compare with another Energy CSV",
            key_prefix="cmp_dissip",
        )

        with st.expander("📘 What do these plots indicate?"):
            st.markdown(dissipation_explanation())

        with st.expander("📗 Energy dissipation analysis for dummies"):
            st.markdown(dissipation_explanation_for_dummies())

        ai_expander("Study Dissipation Analysis", res, uf)

    elif res.kind == "study_modal":
        st.write("#### Modal amplitude summary")
        a_max = df["a"].abs().max()
        a_end = df["a"].iloc[-1]
        summary = pd.DataFrame(
            {
                "max |a(t)|": [a_max],
                "final a(T)": [a_end],
                "relative final amplitude a(T)/max|a|": [a_end / a_max if a_max != 0 else float("nan")],
            }
        )
        st.dataframe(summary, width='stretch')

        st.write("### Modal amplitude and velocity")

        y_cols = [c for c in ["a", "adot"] if c in df.columns]

        fig = plot_compare_timeseries({uf.name: df}, "t", y_cols, title="Modal amplitude and velocity")
        st.plotly_chart(fig, width="stretch")

        compare_expander_timeseries(
            kind="study_modal",
            current_name=uf.name,
            current_df=df,
            x_col="t",
            y_cols=y_cols,
            title="Modal comparison",
            expander_label="🔁 Compare with another Modal CSV",
            key_prefix="cmp_modal",
        )

        with st.expander("📘 What does this plot indicate?"):
            st.markdown(modal_explanation_text())

        with st.expander("📗 Modal amplitude and velocity for dummies"):
            st.markdown(modal_explanation_for_dummies())

        ai_expander("Modal Analysis", res, uf)

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

        x_col = "dt" if "dt" in df_conv.columns else "h"
        y_cols = [c for c in ["u_L2", "u_H1", "v_L2"] if c in df_conv.columns]

        try:
            st.write("### Convergence (log–log)")
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

        fig_simple = plot_compare_timeseries({uf.name: df_conv}, x_col, y_cols, title="Convergence curves")
        st.plotly_chart(fig_simple, width="stretch")

        compare_expander_timeseries(
            kind="conv",
            current_name=uf.name,
            current_df=df_conv,
            x_col=x_col,
            y_cols=y_cols,
            title="Convergence comparison",
            expander_label="🔁 Compare with another Convergence CSV",
            key_prefix="cmp_conv",
        )

        with st.expander("📘 What do these plots indicate?"):
            st.markdown(convergence_explanation())

        with st.expander("📗 Convergence analysis for dummies"):
            st.markdown(convergence_explanation_for_dummies())

        ai_expander("Convergence Analysis", res, uf)

    elif res.kind == "mms":
        metrics = [c for c in df.columns if c.startswith("error_")]
        x = "time" if "time" in df.columns else (
            "t" if "t" in df.columns else ("step" if "step" in df.columns else "n"))
        if metrics:
            # Use last valid sample as a compact "final-time" indicator
            last = df[[x] + metrics].dropna().tail(1)
            if not last.empty:
                st.write("#### Snapshot at the last available sample")
                st.dataframe(last, width='stretch')
            # show peak error over time (helps detect spikes/instabilities)
            st.write("#### Peak error over the run (max over time)")
            peak = df[metrics].max(numeric_only=True).to_frame(name="max").T
            st.dataframe(peak, width='stretch')
        metrics = [c for c in df.columns if c.startswith("error_")]
        x = "time" if "time" in df.columns else ("t" if "t" in df.columns else res.x_candidates[0])
        st.write(f"### MMS error history vs `{x}`")

        # Default: plot current file alone (Plotly = legend toggling)
        fig = plot_compare_timeseries({uf.name: df}, x, metrics, title="MMS error history")
        st.plotly_chart(fig, width="stretch")

        compare_expander_timeseries(
            kind="mms",
            current_name=uf.name,
            current_df=df,
            x_col=x,
            y_cols=metrics,
            title="MMS comparison",
            expander_label="🔁 Compare with another MMS CSV",
            key_prefix="cmp_mms",
        )

        with st.expander("📘 What do these plots indicate?"):
            st.markdown(mms_explanation())

        with st.expander("📗 MMS analysis for dummies"):
            st.markdown(mms_explanation_for_dummies())

        ai_expander("MMS Analysis", res, uf)

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')
