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
) -> None:
    """
    Create an expander in Streamlit to compare time series data from different
    uploaded CSV files.

    Args:
        kind (str): The kind/category of the data to compare.
        current_name (str): The name of the current dataset.
        current_df (pd.DataFrame): The DataFrame of the current dataset.
        x_col (str): The column name to use for the x-axis.
        y_cols (list[str]): A list of column names to plot on the y-axis.
        title (str): The title of the plot.
        expander_label (str): The label for the Streamlit expander.
        key_prefix (str): A prefix for Streamlit widget keys to ensure uniqueness.
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

        fig = plot_compare_timeseries(
            {current_name: current_df, other: other_df},
            x_col,
            selected_y,
            title=title,
        )
        st.plotly_chart(fig, width="stretch")
