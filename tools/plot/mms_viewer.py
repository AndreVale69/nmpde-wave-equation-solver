import io

import json
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="MMS Plot Viewer", layout="wide")

st.title("MMS Plot Viewer")
st.write(
    "Upload one or more CSVs (with columns like `time`, `error_u_L2`, `error_u_H1`, etc.) and compare plots."
)

# Improve sidebar widget visibility while respecting light/dark themes
st.markdown(
        """
        <style>
        /* Use media queries so we don't force a white sidebar in dark mode. */
        @media (prefers-color-scheme: dark) {
            section[data-testid="stSidebar"] div[role="listbox"],
            section[data-testid="stSidebar"] .stMultiSelect,
            section[data-testid="stSidebar"] .stSelectbox,
            section[data-testid="stSidebar"] label {
                color: #fff !important;
                background-color: rgba(255,255,255,0.04) !important;
                border-radius: 4px;
            }
        }
        @media (prefers-color-scheme: light) {
            section[data-testid="stSidebar"] div[role="listbox"],
            section[data-testid="stSidebar"] .stMultiSelect,
            section[data-testid="stSidebar"] .stSelectbox,
            section[data-testid="stSidebar"] label {
                color: #000 !important;
                background-color: rgba(0,0,0,0.03) !important;
                border-radius: 4px;
            }
        }
        /* Small padding tweak for list items for improved readability */
        section[data-testid="stSidebar"] .stMultiSelect > div,
        section[data-testid="stSidebar"] div[role="listbox"] > div {
            padding: 2px 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
)


def _normalize_column_name(c: str) -> str:
    """Best-effort cleanup for messy CSV headers.

    Handles cases like columns literally named "\"                            \"" or with
    leading/trailing whitespace.
    """
    if c is None:
        return ""
    c = str(c)
    # strip whitespace and surrounding quotes
    c = c.strip().strip("\"").strip("'").strip()
    # collapse internal whitespace (e.g. accidental padding)
    c = re.sub(r"\s+", " ", c)
    return c


def _drop_spacer_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns that are empty after normalization.
    drop_cols = []
    rename_map = {}
    for c in df.columns:
        nc = _normalize_column_name(c)
        if nc == "":
            drop_cols.append(c)
        else:
            rename_map[c] = nc
    df = df.rename(columns=rename_map)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert any non-index columns that look numeric.
    # Keep as-is for columns that can't be coerced.
    for c in df.columns:
        if c in {"step"}:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                # leave column unchanged if conversion fails
                pass
            continue
        if c in {"time", "t"} or any(
            c.startswith(p)
            for p in (
                "error_",
                "delta_",
                "rel_delta_",
                "cum_mean_",
                "cum_rms_",
                "h",
            )
        ):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_canonical_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add/alias columns so the rest of the app can rely on canonical names.

    Canonical names (new):
      - error_u_L2, error_u_H1, error_v_L2 (and optionally error_v_H1)
      - delta_u_L2, delta_u_H1, delta_v_L2
      - rel_delta_u_L2, rel_delta_u_H1, rel_delta_v_L2
      - cum_mean_u_L2, cum_mean_u_H1, cum_mean_v_L2
      - cum_rms_u_L2, cum_rms_u_H1, cum_rms_v_L2

    Backward-compatibility:
      - error_u -> error_u_L2
      - error_v -> error_v_L2
      - delta_u -> delta_u_L2
      - delta_v -> delta_v_L2
      - cum_mean_u -> cum_mean_u_L2
      - cum_rms_u  -> cum_rms_u_L2
      - cum_mean_v -> cum_mean_v_L2
      - cum_rms_v  -> cum_rms_v_L2
    """

    alias_pairs = {
        "error_u": "error_u_L2",
        "error_v": "error_v_L2",
        "delta_u": "delta_u_L2",
        "delta_v": "delta_v_L2",
        "rel_delta_u": "rel_delta_u_L2",
        "rel_delta_v": "rel_delta_v_L2",
        "cum_mean_u": "cum_mean_u_L2",
        "cum_rms_u": "cum_rms_u_L2",
        "cum_mean_v": "cum_mean_v_L2",
        "cum_rms_v": "cum_rms_v_L2",
    }
    for old, new in alias_pairs.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # common legacy column name
    if "t" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"t": "time"})

    return df


def _available_metrics(df: pd.DataFrame) -> dict:
    """Return availability of metric families for UI defaults."""
    families = {
        "Instantaneous": [
            "error_u_L2",
            "error_u_H1",
            "error_v_L2",
            "error_v_H1",
        ],
        "Cumulative": [
            "cum_mean_u_L2",
            "cum_rms_u_L2",
            "cum_mean_u_H1",
            "cum_rms_u_H1",
            "cum_mean_v_L2",
            "cum_rms_v_L2",
        ],
        "Delta": [
            "delta_u_L2",
            "delta_u_H1",
            "delta_v_L2",
            "delta_v_H1",
        ],
        "Rel delta": [
            "rel_delta_u_L2",
            "rel_delta_u_H1",
            "rel_delta_v_L2",
            "rel_delta_v_H1",
        ],
    }
    out = {}
    cols = set(df.columns)
    for fam, keys in families.items():
        out[fam] = any(k in cols for k in keys)
    return out


# helper: convert DataFrame (with index) to a Markdown table
def df_with_index_to_markdown(df, index_name="metric", value_format=None):
    cols = list(df.columns)
    header = "| " + index_name + " | " + " | ".join(cols) + " |"
    sep = "| --- " + " | ---" * len(cols) + " |"
    lines = [header, sep]
    for idx in df.index:
        row_vals = []
        for c in cols:
            v = df.at[idx, c]
            if pd.isna(v):
                s = ""
            else:
                if value_format is not None:
                    try:
                        s = value_format(v)
                    except Exception:
                        s = str(v)
                else:
                    s = str(v)
            row_vals.append(s)
        lines.append("| " + str(idx) + " | " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


# allow multiple files
uploaded = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)

if not uploaded:
    st.info("Upload one or more CSV files to get started.")
    st.stop()

# read uploaded files into dict: name -> DataFrame
datasets = {}
read_errors = []
for f in uploaded:
    try:
        df = pd.read_csv(f)
        df = _drop_spacer_columns(df)
        df = _ensure_canonical_schema(df)
        df = _coerce_numeric_columns(df)
        datasets[f.name] = df
    except Exception as e:
        read_errors.append((f.name, str(e)))

if read_errors:
    for name, err in read_errors:
        st.error(f"Failed to read {name}: {err}")

file_names = list(datasets.keys())

if not file_names:
    st.error("No valid CSV files uploaded.")
    st.stop()

# show previews in tabs
tabs = st.tabs(file_names)
for name, tab in zip(file_names, tabs):
    df = datasets[name]
    with tab:
        st.subheader(f"Preview - {name}")
        st.dataframe(df.head(20), width='stretch')
        if st.checkbox(f"Show full dataframe: {name}"):
            st.dataframe(df, width='stretch')

        st.download_button(
            "Download full Markdown",
            data=df_with_index_to_markdown(df, index_name="column"),
            file_name=f"{name}_full.md",
            mime="text/markdown",
        )

# sidebar controls
st.sidebar.header("Files & Plots")
selected = st.sidebar.multiselect("Select files to plot (overlay)", options=file_names, default=file_names[:1])
if not selected:
    st.sidebar.warning("Select at least one file to plot.")
    st.stop()

# required x-axis
for name in selected:
    if "time" not in datasets[name].columns and "step" not in datasets[name].columns:
        st.error(f"File {name} must contain a `time` or `step` column.")
        st.stop()

# Decide x-axis (time preferred)
x_axis = st.sidebar.selectbox("X axis", options=["time", "step"], index=0)
# If any selected file lacks time, fall back to step automatically.
if x_axis == "time" and any("time" not in datasets[n].columns for n in selected):
    x_axis = "step"
    st.sidebar.info("Some files are missing `time`; using `step` on the x-axis.")

# sidebar plot toggles
show_inst = st.sidebar.checkbox("Instantaneous errors (log scale)", value=True)
show_delta = st.sidebar.checkbox("Step-to-step deltas", value=False)
show_reldelta = st.sidebar.checkbox("Relative step-to-step deltas", value=False)
show_table = st.sidebar.checkbox("Show comparison table", value=False)
show_conv = st.sidebar.checkbox("Show convergence (error vs h, log-log)", value=False)
legend_mode = st.sidebar.selectbox("Legend placement", options=["Outside (compact)", "Inside (default)"], index=0)

# Theming & palettes (Light theme removed)
palette_choice = st.sidebar.selectbox("Color palette", options=["Default", "High contrast", "Colorblind-friendly"], index=0)

# palette definitions
palettes = {
    "Default": px.colors.qualitative.Plotly,
    "High contrast": ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"],
    "Colorblind-friendly": px.colors.qualitative.Set1,
}
palette = palettes.get(palette_choice, px.colors.qualitative.Plotly)

# session-state: aliases and presets
if "aliases" not in st.session_state:
    st.session_state.aliases = {name: name for name in file_names}

st.sidebar.markdown("---")
st.sidebar.subheader("File aliases")
for name in file_names:
    # create a stable widget key using filename
    key = f"alias__{name}"
    val = st.sidebar.text_input(f"Alias for {name}", value=st.session_state.aliases.get(name, name), key=key)
    st.session_state.aliases[name] = val

st.sidebar.markdown("---")
st.sidebar.subheader("Metric selection")

# Detect which norms are available across selected files, so users can actually find H1.
# We consider H1 available if any key H1 column exists in any selected dataset.

def _has_any_col(name: str, cols: set[str]) -> bool:
    return name in cols


def _norm_available_in_any_selected(norm: str) -> bool:
    keys = [
        f"error_u_{norm}",
        f"error_v_{norm}",
        f"cum_mean_u_{norm}",
        f"cum_rms_u_{norm}",
        f"delta_u_{norm}",
        f"rel_delta_u_{norm}",
    ]
    for n in selected:
        cols = set(datasets[n].columns)
        if any(k in cols for k in keys):
            return True
    return False


available_norms = [n for n in ("L2", "H1") if _norm_available_in_any_selected(n)]
if not available_norms:
    available_norms = ["L2", "H1"]

metric_norm = st.sidebar.selectbox("Norm", options=available_norms, index=0)

if "H1" not in available_norms:
    st.sidebar.warning(
        "No H1 columns detected in the selected file(s). "
        "To see H1 plots, the CSV must contain columns like `error_u_H1`, `delta_u_H1`, `cum_mean_u_H1`, ..."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Presets")
preset_file = st.sidebar.file_uploader("Load preset (JSON)", type=["json"])
if preset_file is not None:
    try:
        pdata = json.load(preset_file)
        # only set values that exist in the preset
        if "aliases" in pdata:
            for k, v in pdata["aliases"].items():
                if k in st.session_state.aliases:
                    st.session_state.aliases[k] = v
        if "selected" in pdata:
            # update selected via session state trick
            st.session_state["_preset_selected"] = pdata["selected"]
        if "legend_mode" in pdata:
            st.session_state["_preset_legend_mode"] = pdata["legend_mode"]
        if "show_inst" in pdata:
            st.session_state["_preset_show_inst"] = pdata["show_inst"]
        if "show_delta" in pdata:
            st.session_state["_preset_show_delta"] = pdata["show_delta"]
        if "show_reldelta" in pdata:
            st.session_state["_preset_show_reldelta"] = pdata.get("show_reldelta", show_reldelta)
        if "metric_norm" in pdata:
            st.session_state["_preset_metric_norm"] = pdata["metric_norm"]
        if "x_axis" in pdata:
            st.session_state["_preset_x_axis"] = pdata["x_axis"]

        selected = st.session_state.get("_preset_selected", selected)
        legend_mode = st.session_state.get("_preset_legend_mode", legend_mode)
        show_inst = st.session_state.get("_preset_show_inst", show_inst)
        show_delta = st.session_state.get("_preset_show_delta", show_delta)
        show_reldelta = st.session_state.get("_preset_show_reldelta", show_reldelta)
        metric_norm = st.session_state.get("_preset_metric_norm", metric_norm)
        x_axis = st.session_state.get("_preset_x_axis", x_axis)

        st.sidebar.success("Preset loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load preset: {e}")

# expose save preset button
if st.sidebar.button("Save current preset"):
    preset = {
        "aliases": st.session_state.aliases,
        "selected": selected,
        "legend_mode": legend_mode,
        "show_inst": show_inst,
        "show_delta": show_delta,
        "show_reldelta": show_reldelta,
        "metric_norm": metric_norm,
        "x_axis": x_axis,
    }
    st.sidebar.download_button(
        "Download preset.json",
        data=json.dumps(preset, indent=2),
        file_name="mms_preset.json",
        mime="application/json",
    )


def _get_x(df: pd.DataFrame, x_axis: str):
    if x_axis in df.columns:
        return df[x_axis]
    # fallback
    if "time" in df.columns:
        return df["time"]
    return df.index


def _col(metric: str, var: str, norm: str):
    # metric: error / delta / rel_delta / cum_mean / cum_rms
    return f"{metric}_{var}_{norm}"


def _help_block(title: str, body_md: str):
    """Reusable help expander used under each plot section."""
    with st.expander(f"Help: {title}", expanded=False):
        st.markdown(body_md)


def _norm_cols_hint(norm: str) -> str:
    return (
        f"- Instantaneous: `error_u_{norm}`, `error_v_{norm}`\n"
        f"- Step delta: `delta_u_{norm}`, `delta_v_{norm}`\n"
        f"- Relative delta: `rel_delta_u_{norm}`, `rel_delta_v_{norm}`\n"
        f"- Cumulative: `cum_mean_u_{norm}`, `cum_rms_u_{norm}`, `cum_mean_v_{norm}`, `cum_rms_v_{norm}`\n"
    )


# Sidebar: comparison table toggle
if show_table:
    _help_block(
        "Comparison table",
        """
**What this is**  
A numeric summary of the instantaneous error series for the *selected norm* (L2/H1). It’s useful for quickly comparing runs without visually inspecting every plot.

**What it computes (per file and variable)**  
- min / max / mean / median / stddev  
- RMS (root-mean-square)  
- final value (last sample)  
- `x_of_max` / `x_of_min` (time or step where the extreme happens)  
- `max_rel_step_change` = max over steps of `|e_n − e_{n-1}| / |e_{n-1}|` (ignores divisions by 0)

**Columns used**  
It pulls from:  
- `error_u_<NORM>` and/or `error_v_<NORM>`

**Tips / troubleshooting**  
- If the table is empty, your CSV likely doesn’t contain the chosen norm (e.g. missing `error_u_H1`). Switch **Norm** in the sidebar or check the CSV header.  
- If you don’t have a `time` column, choose **X axis = step** (we’ll still compute `x_of_*`).
""",
    )

    # build stats table for selected files
    stats_names = [
        "min",
        "max",
        "mean",
        "median",
        "stddev",
        "rms",
        "final",
        "x_of_max",
        "x_of_min",
        "max_rel_step_change",
    ]

    def compute_stats(x, series):
        if series is None or series.dropna().empty:
            return {k: np.nan for k in stats_names}
        s = series.dropna().astype(float)
        out = {}
        out["min"] = s.min()
        out["max"] = s.max()
        out["mean"] = s.mean()
        out["median"] = s.median()
        out["stddev"] = s.std()
        out["rms"] = np.sqrt(np.mean(s.values ** 2))
        out["final"] = s.iloc[-1]
        # x of extrema (time/step)
        try:
            idx_max = s.idxmax()
            out["x_of_max"] = float(x.loc[idx_max]) if x is not None else np.nan
        except Exception:
            out["x_of_max"] = np.nan
        try:
            idx_min = s.idxmin()
            out["x_of_min"] = float(x.loc[idx_min]) if x is not None else np.nan
        except Exception:
            out["x_of_min"] = np.nan
        # max relative step change
        prev = s.shift(1)
        denom = prev.abs()
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (s - prev).abs() / denom
        rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
        out["max_rel_step_change"] = rel.max() if not rel.empty else np.nan
        return out

    cols = {}
    for name in selected:
        df = datasets[name]
        alias = st.session_state.aliases.get(name, name)
        x = _get_x(df, x_axis)

        # u and v, chosen norm
        for var in ("u", "v"):
            c = _col("error", var, metric_norm)
            if c in df.columns:
                stats = compute_stats(x, df[c])
                cols[f"{alias}: {var} ({metric_norm})"] = [stats.get(k, np.nan) for k in stats_names]

    if not cols:
        st.warning("No matching error columns found for the selected norm.")
    else:
        stats_df = pd.DataFrame(cols, index=stats_names)
        stats_display = stats_df.copy()
        for c in stats_display.columns:
            stats_display[c] = stats_display[c].map(lambda x: "{:.5e}".format(x) if pd.notna(x) else "")

        st.subheader("Comparison table: summary statistics")
        st.caption(f"Based on `{x_axis}` and `{metric_norm}` instantaneous errors.")
        st.dataframe(stats_display.style.set_table_attributes('style="font-family: monospace;"'))

        md_full = df_with_index_to_markdown(
            stats_df,
            index_name="stat",
            value_format=lambda x: "{:.5e}".format(x) if pd.notna(x) else "",
        )
        st.download_button(
            "Download comparison Markdown (full)",
            data=md_full,
            file_name="comparison_stats_full.md",
            mime="text/markdown",
        )


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


def fig_to_pdf_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    buf.seek(0)
    return buf


col1, col2 = st.columns(2)

# helper: style cycle (plotly will assign colors automatically)
markers = ["circle", "square", "diamond", "triangle-up", "triangle-down", "x", "cross", "star"]
lines = ["solid", "dash", "dot", "dashdot"]


# helper for safe image export (handles Kaleido/Chrome missing)
def safe_image_bytes(fig, fmt="png", scale=2):
    try:
        img = fig.to_image(format=fmt, scale=scale)
        return img, None
    except Exception as e:
        return None, str(e)


# ---------- CONVERGENCE: error vs h (log-log) ----------
def extract_h_from_name(name: str):
    # try patterns like h=0.01 or _h0.01 or -h0.01
    m = re.search(r"h=?([0-9.eE+-]+)", name)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


if show_conv:
    _help_block(
        "Convergence (error vs h)",
        """
**Goal**  
Estimate the empirical convergence order by fitting a straight line in log–log scale:
$$
\\log_{10}(e) \\approx m\,\\log_{10}(h) + c
$$

The slope `m` is the observed order.

**What you need in the data**  
- A mesh-size indicator `h`, either as a column named `h` **or** encoded in the filename (patterns like `h=0.01`, `_h0.01`, `-h0.01`).
- An error column for the chosen norm: `error_u_<NORM>` and/or `error_v_<NORM>`.

**What the plot shows**  
- Marker points: `(h, error)` extracted from each selected file (uses the *last* available error sample).  
- A fitted line (least squares on log10) plus the slope in the legend.

**Interpretation tips**  
- Make sure you’re in the asymptotic regime (enough refinement).  
- If errors are ~0 or negative (can happen with bad parsing), those points are dropped.

**Troubleshooting**  
- "Not enough data..." usually means fewer than 2 valid `(h, error)` points. Add more refinement levels or provide `h`.
""",
    )

    # gather (h, error) for selected datasets
    hu = []
    eu = []
    hv = []
    ev = []
    for name in selected:
        df = datasets[name]
        hval = None
        if "h" in df.columns:
            try:
                hval = float(df["h"].dropna().iloc[-1])
            except Exception:
                hval = None
        if hval is None:
            hval = extract_h_from_name(name)
        if hval is None:
            continue

        cu = _col("error", "u", metric_norm)
        cv = _col("error", "v", metric_norm)
        if cu in df.columns:
            try:
                err = float(df[cu].dropna().iloc[-1])
                hu.append(hval)
                eu.append(err)
            except Exception:
                pass
        if cv in df.columns:
            try:
                err = float(df[cv].dropna().iloc[-1])
                hv.append(hval)
                ev.append(err)
            except Exception:
                pass

    conv_fig = go.Figure()

    def add_loglog_series(hs, es, label):
        if len(hs) < 2:
            return None
        hs = np.array(hs)
        es = np.array(es)
        mask = (hs > 0) & (es > 0)
        if mask.sum() < 2:
            return None
        hs = hs[mask]
        es = es[mask]
        m, c = np.polyfit(np.log10(hs), np.log10(es), 1)
        hsort = np.sort(hs)
        fit = 10 ** (m * np.log10(hsort) + c)
        conv_fig.add_trace(go.Scatter(x=hs, y=es, mode="markers", name=f"{label} points"))
        conv_fig.add_trace(go.Scatter(x=hsort, y=fit, mode="lines", name=f"{label} fit (slope={m:.2f})"))
        return m

    slope_u = add_loglog_series(hu, eu, f"u ({metric_norm})")
    slope_v = add_loglog_series(hv, ev, f"v ({metric_norm})")
    if slope_u is None and slope_v is None:
        st.warning(
            "Not enough data with 'h' and matching error columns to compute convergence slopes. "
            "Provide an 'h' column or include h in filenames."
        )
    else:
        conv_fig.update_xaxes(title_text="h", type="log")
        conv_fig.update_yaxes(title_text=f"Error ({metric_norm})", type="log")
        st.subheader("Convergence: error vs h (log-log)")
        st.plotly_chart(conv_fig, use_container_width=True)
        if slope_u is not None:
            st.write(f"Estimated slope for u ({metric_norm}): {slope_u:.3f}")
        if slope_v is not None:
            st.write(f"Estimated slope for v ({metric_norm}): {slope_v:.3f}")


# ---------- FIG 1: instantaneous errors (interactive) ----------
if show_inst:
    st.subheader("Instantaneous MMS error - comparison (interactive)")
    _help_block(
        "Instantaneous MMS error",
        f"""
**What this plot is**  
The per-time-step error of your numerical solution against the manufactured solution.

**Columns used (chosen Norm = `{metric_norm}`)**  
- `error_u_{metric_norm}` for displacement/solution `u`  
- `error_v_{metric_norm}` for velocity `v` (if present)

**Axis meaning**  
- X axis: `{x_axis}` (choose in sidebar; `time` is preferred, `step` works too)  
- Y axis: error magnitude in **log scale** (so you can see decades of change)

**How to read it**  
- A downward trend typically means the method is stable/accurate for the configuration.  
- Oscillations can be physical (wave-like) or numerical (time integration artifacts).  
- Sudden spikes can indicate CFL/time-step issues, solver divergence, or boundary/forcing discontinuities.

**Common gotchas**  
- If you don’t see H1: your CSV must contain `error_u_H1`/`error_v_H1`. The sidebar will hide `H1` if it isn’t detected.
- If only `u` appears: the file likely doesn’t contain the corresponding `v` column.
""",
    )

    fig = go.Figure()

    any_u = False
    any_v = False
    for i, name in enumerate(selected):
        df = datasets[name]
        x = _get_x(df, x_axis)
        alias = st.session_state.aliases.get(name, name)
        marker_sym = markers[i % len(markers)]
        dash = lines[i % len(lines)]
        color = palette[i % len(palette)]
        marker_outline = "#ffffff"

        cu = _col("error", "u", metric_norm)
        cv = _col("error", "v", metric_norm)
        if cu in df.columns:
            any_u = True
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[cu],
                    mode="lines+markers",
                    name=f"{alias}: u ({metric_norm})",
                    marker=dict(symbol=marker_sym, color=color, line=dict(color=marker_outline, width=1)),
                    line=dict(color=color, dash=dash, width=2),
                )
            )
        if cv in df.columns:
            any_v = True
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[cv],
                    mode="lines+markers",
                    name=f"{alias}: v ({metric_norm})",
                    marker=dict(symbol=marker_sym, color=color, line=dict(color=marker_outline, width=1)),
                    line=dict(color=color, dash=dash, width=2),
                )
            )

    if not any_u and not any_v:
        st.warning(f"None of the selected files contain instantaneous `{metric_norm}` error columns.")

    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text=f"{metric_norm} error", type="log")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.download_button(
        "Download instantaneous series JSON",
        data=fig.to_json(),
        file_name="instantaneous_plot.json",
        mime="application/json",
    )


# ---------- FIG 3: deltas (interactive) ----------
if show_delta:
    st.subheader("Step-to-step variation - comparison (interactive)")
    _help_block(
        "Step-to-step deltas",
        f"""
**What this plot is**  
The absolute change of the error from one sample to the next. It helps you spot bursts/instabilities.

Typically:
$$
\\Delta e_n = e_n - e_{{n-1}}
$$

(Some codes store an absolute value; this viewer simply plots what’s in the CSV.)

**Columns used (chosen Norm = `{metric_norm}`)**  
- `delta_u_{metric_norm}`, `delta_v_{metric_norm}`

**How to read it**  
- Values near 0 mean the error is evolving smoothly.  
- Large magnitude excursions mean a sudden change in error (often aligned with sharp forcing, reflections, or a too-large dt).

**Troubleshooting**  
- If you don’t see lines, that usually means the column isn’t present (e.g. your CSV has only `delta_*_L2`). Switch Norm or regenerate the CSV.
""",
    )

    fig = go.Figure()

    any_delta = False
    for i, name in enumerate(selected):
        df = datasets[name]
        x = _get_x(df, x_axis)
        alias = st.session_state.aliases.get(name, name)
        dash = lines[i % len(lines)]
        color = palette[i % len(palette)]

        for var in ("u", "v"):
            c = _col("delta", var, metric_norm)
            if c in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[c],
                        mode="lines",
                        name=f"{alias}: Δ{var} ({metric_norm})",
                        line=dict(color=color, dash=dash, width=2),
                    )
                )
                any_delta = True

    if not any_delta:
        st.warning("None of the selected files contain delta columns for the chosen norm.")

    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text="Increment")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.download_button(
        "Download deltas series JSON",
        data=fig.to_json(),
        file_name="deltas_plot.json",
        mime="application/json",
    )


# ---------- FIG 4: relative deltas (interactive) ----------
if show_reldelta:
    st.subheader("Relative step-to-step variation - comparison (interactive)")
    _help_block(
        "Relative step-to-step deltas",
        f"""
**What this plot is**  
The step-to-step change scaled by the previous value, typically:
$$
r_n = \\frac{{|e_n - e_{{n-1}}|}}{{|e_{{n-1}}|}}
$$

This is a dimensionless "percent-like" change indicator.

**Columns used (chosen Norm = `{metric_norm}`)**  
- `rel_delta_u_{metric_norm}`, `rel_delta_v_{metric_norm}`

**How to read it**  
- Near 0: stable evolution.  
- Large spikes: abrupt transitions or times where the denominator is small.

**Important gotcha**  
- If the previous error is ~0, the relative delta can blow up or become undefined. Your CSV generator may clamp/skip; the viewer will just plot what’s provided.
""",
    )

    fig = go.Figure()

    any_delta = False
    for i, name in enumerate(selected):
        df = datasets[name]
        x = _get_x(df, x_axis)
        alias = st.session_state.aliases.get(name, name)
        dash = lines[i % len(lines)]
        color = palette[i % len(palette)]

        for var in ("u", "v"):
            c = _col("rel_delta", var, metric_norm)
            if c in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[c],
                        mode="lines",
                        name=f"{alias}: relΔ{var} ({metric_norm})",
                        line=dict(color=color, dash=dash, width=2),
                    )
                )
                any_delta = True

    if not any_delta:
        st.warning("None of the selected files contain relative-delta columns for the chosen norm.")

    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text="Relative increment")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.download_button(
        "Download relative deltas series JSON",
        data=fig.to_json(),
        file_name="rel_deltas_plot.json",
        mime="application/json",
    )
