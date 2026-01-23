from typing import Callable, Iterable, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def plot_compare_timeseries(
        dfs: dict[str, pd.DataFrame],
        x_col: str,
        y_cols: list[str],
        title: str | None = None
) -> go.Figure:
    """
    Plot multiple time series from different DataFrames for comparison.

    Args:
        dfs (dict[str, pd.DataFrame]): A dictionary mapping names to DataFrames.
        x_col (str): The column name to use for the x-axis.
        y_cols (list[str]): A list of column names to plot on the y-axis.
        title (str): The title of the plot.
    Returns:
        go.Figure: The resulting Plotly figure.
    """
    fig = go.Figure()

    for name, d in dfs.items():
        # only keep rows where x & all y exist
        dd = d[[x_col] + y_cols].dropna()
        if dd.empty:
            continue
        for y in y_cols:
            fig.add_trace(
                go.Scatter(
                    x=dd[x_col],
                    y=dd[y],
                    mode="lines",
                    name=f"{name} · {y}",
                )
            )

    if title is not None:
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="value",
            legend_title="Click to hide/show",
            hovermode="x unified",
        )
    else:
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title="value",
            legend_title="Click to hide/show",
            hovermode="x unified",
        )
    return fig


def compare_expander_timeseries(
        *,
        kind: str,
        current_name: str,
        current_df: pd.DataFrame,
        x_col: str,
        y_cols: list[str],
        title: str,
        expander_label: str,
        key_prefix: str,
        # AI comparison options
        enable_ai: bool = False,
        build_comparison_prompt: Optional[Callable[[str, pd.DataFrame, str, pd.DataFrame, str], str]] = None,
        stream_generate: Optional[Callable[[str, str, str], Iterable[str]]] = None,
        default_host: str = "https://ollama.com",
        default_model: str = "gpt-oss:120b-cloud",
) -> None:
    """
    Streamlit expander to compare the current time series DataFrame against others of the same kind.
    Args:
        kind (str): The kind of analysis (e.g., "time_series").
        current_name (str): The name of the current DataFrame/run.
        current_df (pd.DataFrame): The current DataFrame to compare from.
        x_col (str): The column name to use for the x-axis.
        y_cols (list[str]): A list of column names to plot on the y-axis.
        title (str): The title of the plot.
        expander_label (str): The label for the Streamlit expander.
        key_prefix (str): A prefix for Streamlit widget keys to avoid collisions.
        enable_ai (bool): Whether to enable AI comparison features.
        build_comparison_prompt (Callable): Function to build the AI prompt for comparison.
        stream_generate (Callable): Function to stream AI-generated text.
        default_host (str): Default host for the AI model.
        default_model (str): Default model name for the AI.
    """
    with st.expander(expander_label):
        pool = st.session_state.get("parsed_by_kind", {}).get(kind, [])
        candidates = [e["name"] for e in pool if e["name"] != current_name]

        if not candidates:
            st.info("No other CSV uploaded to compare.")
            return

        other = st.selectbox(
            "Compare against",
            ["(none)"] + candidates,
            key=f"{key_prefix}_other_{current_name}",
        )

        selected_y = st.multiselect(
            "Series to show",
            options=y_cols,
            default=y_cols,
            key=f"{key_prefix}_series_{current_name}",
        )

        if other == "(none)" or not selected_y:
            return

        other_df = next(e["df"] for e in pool if e["name"] == other)

        # Plot overlay
        fig = plot_compare_timeseries(
            {current_name: current_df, other: other_df},
            x_col,
            selected_y,
            title=title,
        )
        st.plotly_chart(fig, width="stretch")

        # -------------------------
        # Optional AI comparison
        # -------------------------
        if enable_ai and build_comparison_prompt and stream_generate:
            with st.expander("🤖 AI comparison comment"):
                host = st.text_input(
                    "Ollama host",
                    value=default_host,
                    key=f"{key_prefix}_ai_host_{current_name}",
                )
                model = st.text_input(
                    "Model",
                    value=default_model,
                    key=f"{key_prefix}_ai_model_{current_name}",
                )

                # (Optional) show help link
                st.caption("Models: https://ollama.com/search")

                if st.button(
                        "Generate AI comparison",
                        key=f"{key_prefix}_ai_btn_{current_name}",
                        type="secondary",
                ):
                    try:
                        with st.spinner("🤖 Generating comparison...", show_time=True):
                            prompt = build_comparison_prompt(
                                kind, current_df, current_name, other_df, other
                            )

                            out = st.empty()
                            acc = ""
                            for chunk in stream_generate(prompt, model, host):
                                acc += chunk
                                out.markdown(acc)
                        out.markdown(acc)
                    except Exception as e:
                        st.error(f"Ollama error: {e}")
