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
st.write("Upload one or more CSVs (with columns like `time`, `error_u`, `error_v`, etc.) and compare plots.")

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
        datasets[f.name] = pd.read_csv(f)
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

# sidebar controls
st.sidebar.header("Files & Plots")
selected = st.sidebar.multiselect("Select files to plot (overlay)", options=file_names, default=file_names[:1])
if not selected:
    st.sidebar.warning("Select at least one file to plot.")
    st.stop()

required_cols = {"time"}
for name in selected:
    if not required_cols.issubset(datasets[name].columns):
        st.error(f"File {name} must contain a `time` column.")
        st.stop()

# sidebar plot toggles
show_inst = st.sidebar.checkbox("Instantaneous errors (log scale)", value=True)
show_cum = st.sidebar.checkbox("Cumulative statistics", value=True)
show_delta = st.sidebar.checkbox("Step-to-step deltas", value=False)
show_table = st.sidebar.checkbox("Show comparison table (u / v)", value=False)
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
        if "show_cum" in pdata:
            st.session_state["_preset_show_cum"] = pdata["show_cum"]
        if "show_delta" in pdata:
            st.session_state["_preset_show_delta"] = pdata["show_delta"]
        # st.experimental_rerun()
        # experimental rerun is flaky, so we manually set the variables instead
        selected = st.session_state.get("_preset_selected", selected)
        legend_mode = st.session_state.get("_preset_legend_mode", legend_mode)
        show_inst = st.session_state.get("_preset_show_inst", show_inst)
        show_cum = st.session_state.get("_preset_show_cum", show_cum)
        show_delta = st.session_state.get("_preset_show_delta", show_delta)
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
        "show_cum": show_cum,
        "show_delta": show_delta,
    }
    st.sidebar.download_button("Download preset.json", data=json.dumps(preset, indent=2), file_name="mms_preset.json", mime="application/json")

# Sidebar: comparison table toggle
if show_table:
    # build stats table for selected files
    stats_names = [
        "min",
        "max",
        "mean",
        "median",
        "stddev",
        "rms",
        "final",
        "time_of_max",
        "time_of_min",
        "max_rel_step_change",
    ]
    cols = {}
    for name in selected:
        df = datasets[name]
        alias = st.session_state.aliases.get(name, name)
        t = df.get("time")

        def compute_stats(series):
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
            # times
            try:
                idx_max = s.idxmax()
                out["time_of_max"] = float(t.iloc[idx_max]) if t is not None else np.nan
            except Exception:
                out["time_of_max"] = np.nan
            try:
                idx_min = s.idxmin()
                out["time_of_min"] = float(t.iloc[idx_min]) if t is not None else np.nan
            except Exception:
                out["time_of_min"] = np.nan
            # max relative step change
            prev = s.shift(1)
            denom = prev.abs()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = (s - prev).abs() / denom
            rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
            out["max_rel_step_change"] = rel.max() if not rel.empty else np.nan
            return out

        # u
        stats_u = compute_stats(df.get("error_u"))
        col_u = f"{alias}: u"
        cols[col_u] = [stats_u.get(k, np.nan) for k in stats_names]

        # v
        stats_v = compute_stats(df.get("error_v"))
        col_v = f"{alias}: v"
        cols[col_v] = [stats_v.get(k, np.nan) for k in stats_names]

    stats_df = pd.DataFrame(cols, index=stats_names)
    # format numbers for readability
    stats_display = stats_df.copy()
    for c in stats_display.columns:
        stats_display[c] = stats_display[c].map(lambda x: "{:.5e}".format(x) if pd.notna(x) else "")

    st.subheader("Comparison table: summary statistics")
    st.dataframe(stats_display.style.set_table_attributes('style="font-family: monospace;"'))
    st.download_button("Download comparison CSV", data=stats_df.to_csv(), file_name="comparison_stats.csv", mime="text/csv")

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
                # prefer last h value or median
                hval = float(df["h"].dropna().iloc[-1])
            except Exception:
                hval = None
        if hval is None:
            hval = extract_h_from_name(name)
        if hval is None:
            continue
        # choose representative error: last available error_u or error_v
        if "error_u" in df.columns:
            try:
                err = float(df["error_u"].dropna().iloc[-1])
                hu.append(hval)
                eu.append(err)
            except Exception:
                pass
        if "error_v" in df.columns:
            try:
                err = float(df["error_v"].dropna().iloc[-1])
                hv.append(hval)
                ev.append(err)
            except Exception:
                pass

    conv_fig = go.Figure()
    slopes = {}
    def add_loglog_series(hs, es, label):
        if len(hs) < 2:
            return None
        hs = np.array(hs)
        es = np.array(es)
        # remove non-positive
        mask = (hs > 0) & (es > 0)
        if mask.sum() < 2:
            return None
        hs = hs[mask]
        es = es[mask]
        m, c = np.polyfit(np.log10(hs), np.log10(es), 1)
        # fit line for plotting
        hsort = np.sort(hs)
        fit = 10 ** (m * np.log10(hsort) + c)
        conv_fig.add_trace(go.Scatter(x=hs, y=es, mode="markers", name=f"{label} points"))
        conv_fig.add_trace(go.Scatter(x=hsort, y=fit, mode="lines", name=f"{label} fit (slope={m:.2f})"))
        return m

    slope_u = add_loglog_series(hu, eu, "u")
    slope_v = add_loglog_series(hv, ev, "v")
    if slope_u is None and slope_v is None:
        st.warning("Not enough data with 'h' and error columns to compute convergence slopes. Provide 'h' column or include h in filenames.")
    else:
        conv_fig.update_xaxes(title_text="h", type="log")
        conv_fig.update_yaxes(title_text="Error", type="log")
        st.subheader("Convergence: error vs h (log-log)")
        st.plotly_chart(conv_fig, use_container_width=True)
        if slope_u is not None:
            st.write(f"Estimated slope for u: {slope_u:.3f}")
        if slope_v is not None:
            st.write(f"Estimated slope for v: {slope_v:.3f}")

# ---------- FIG 1: instantaneous errors (interactive) ----------
if show_inst:
    st.subheader("Instantaneous MMS error - comparison (interactive)")
    fig = go.Figure()

    any_u = False
    any_v = False
    for i, name in enumerate(selected):
        df = datasets[name]
        t = df["time"]
        alias = st.session_state.aliases.get(name, name)
        marker_sym = markers[i % len(markers)]
        dash = lines[i % len(lines)]
        color = palette[i % len(palette)]
        # marker outline color for contrast
        marker_outline = "#ffffff"
        if "error_u" in df.columns:
            any_u = True
            fig.add_trace(go.Scatter(
                x=t,
                y=df["error_u"],
                mode="lines+markers",
                name=f"{alias}: u",
                marker=dict(symbol=marker_sym, color=color, line=dict(color=marker_outline, width=1)),
                line=dict(color=color, dash=dash, width=2),
            ))
        if "error_v" in df.columns:
            any_v = True
            fig.add_trace(go.Scatter(
                x=t,
                y=df["error_v"],
                mode="lines+markers",
                name=f"{alias}: v",
                marker=dict(symbol=marker_sym, color=color, line=dict(color=marker_outline, width=1)),
                line=dict(color=color, dash=dash, width=2),
            ))

    if not any_u and not any_v:
        st.warning("None of the selected files contain `error_u` or `error_v` columns.")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="L2 error", type="log")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.download_button("Download instantaneous series JSON", data=fig.to_json(), file_name="instantaneous_plot.json", mime="application/json")


# ---------- FIG 2: cumulative stats (interactive) ----------
if show_cum:
    st.subheader("Cumulative error statistics - comparison (interactive)")
    fig = go.Figure()

    plotted = False
    for i, name in enumerate(selected):
        df = datasets[name]
        t = df["time"]
        alias = st.session_state.aliases.get(name, name)
        dash = lines[i % len(lines)]
        color = palette[i % len(palette)]
        if "cum_mean_u" in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df["cum_mean_u"], mode="lines", name=f"{alias}: Mean(u)", line=dict(color=color, dash=dash, width=2)))
            plotted = True
        if "cum_rms_u" in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df["cum_rms_u"], mode="lines", name=f"{alias}: RMS(u)", line=dict(color=color, dash=dash, width=2)))
            plotted = True
        if "cum_mean_v" in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df["cum_mean_v"], mode="lines", name=f"{alias}: Mean(v)", line=dict(color=color, dash=dash, width=2)))
            plotted = True
        if "cum_rms_v" in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df["cum_rms_v"], mode="lines", name=f"{alias}: RMS(v)", line=dict(color=color, dash=dash, width=2)))
            plotted = True

    if not plotted:
        st.warning("None of the selected files contain cumulative statistic columns (cum_mean_*, cum_rms_*).")

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Error magnitude")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.download_button("Download cumulative series JSON", data=fig.to_json(), file_name="cumulative_plot.json", mime="application/json")

# ---------- FIG 3: deltas (interactive) ----------
if show_delta:
    st.subheader("Step-to-step variation - comparison (interactive)")
    fig = go.Figure()

    any_delta = False
    for i, name in enumerate(selected):
        df = datasets[name]
        t = df["time"]
        alias = st.session_state.aliases.get(name, name)
        dash = lines[i % len(lines)]
        color = palette[i % len(palette)]
        if "delta_u" in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df["delta_u"], mode="lines", name=f"{alias}: delta u$", line=dict(color=color, dash=dash, width=2)))
            any_delta = True
        if "delta_v" in df.columns:
            fig.add_trace(go.Scatter(x=t, y=df["delta_v"], mode="lines", name=f"{alias}: delta v$", line=dict(color=color, dash=dash, width=2)))
            any_delta = True

    if not any_delta:
        st.warning("None of the selected files contain `delta_u` or `delta_v` columns.")

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Increment")
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, width='stretch')

    st.download_button("Download deltas series JSON", data=fig.to_json(), file_name="deltas_plot.json", mime="application/json")
