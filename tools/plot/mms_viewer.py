import hashlib
import io
import re
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    # When running from repo root
    from tools.plot.csv_utils import read_csv_robust
    from tools.plot.convergence_utils import (
        compute_observed_orders,
        detect_kind,
        load_convergence_csv,
        pretty_metric_name,
        summarize_fits,
    )
except Exception:  # pragma: no cover
    # When running from within tools/plot
    from csv_utils import read_csv_robust
    from convergence_utils import (
        compute_observed_orders,
        detect_kind,
        load_convergence_csv,
        pretty_metric_name,
        summarize_fits,
    )


@dataclass(frozen=True)
class _CachedUpload:
    name: str
    content_hash: str
    data: bytes


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mode_cache_key(mode: str) -> str:
    mode = str(mode)
    if mode.lower() in {"mms"}:
        return "uploads_mms"
    if mode.lower() in {"convergence", "conv"}:
        return "uploads_conv"
    if mode.lower() in {"studies", "study"}:
        return "uploads_studies"
    raise ValueError(f"Unknown mode: {mode}")


def _library_add(name: str, bs: bytes) -> _CachedUpload:
    """Add bytes to the global in-session library, returning the canonical cached object."""
    h = _sha256_bytes(bs)
    if h not in st.session_state["uploads_library"]:
        st.session_state["uploads_library"][h] = _CachedUpload(name=name, content_hash=h, data=bs)
    # Track display name (first seen wins; user can still alias in each mode)
    st.session_state["uploads_library_display"].setdefault(h, name)
    return st.session_state["uploads_library"][h]


def _sync_mode_cache_from_library(mode: str):
    """Ensure a mode cache only contains hashes that still exist in the library."""
    key = _mode_cache_key(mode)
    cache = st.session_state.get(key)
    if not cache:
        return
    lib = st.session_state.get("uploads_library", {})
    drop = [h for h in cache.keys() if h not in lib]
    for h in drop:
        cache.pop(h, None)


def _copy_hashes_to_mode(hashes: list[str], *, mode: str):
    key = _mode_cache_key(mode)
    st.session_state.setdefault(key, {})
    lib = st.session_state.get("uploads_library", {})
    for h in hashes:
        cu = lib.get(h)
        if cu is None:
            continue
        st.session_state[key].setdefault(h, cu)


def _remove_hashes_from_mode(hashes: list[str], *, mode: str):
    key = _mode_cache_key(mode)
    cache = st.session_state.get(key, {})
    for h in hashes:
        cache.pop(h, None)


def _move_hashes(hashes: list[str], *, src_mode: str, dst_mode: str):
    """Move hashes between two modes (copy to dst, remove from src)."""
    _copy_hashes_to_mode(hashes, mode=dst_mode)
    _remove_hashes_from_mode(hashes, mode=src_mode)


def _prune_cache_by_filenames(cache_key: str, keep_names: set[str]):
    """Drop cached uploads whose filename is not currently present in the uploader."""
    cache = st.session_state.get(cache_key, {})
    if not cache:
        return
    drop_hashes = [h for h, cu in cache.items() if cu.name not in keep_names]
    for h in drop_hashes:
        cache.pop(h, None)


def _autoselect_new_files(selected_key: str, available: list[str], newly_added: set[str]):
    """Update a session_state selection list by adding newly uploaded files."""
    current = st.session_state.get(selected_key)
    if current is None:
        # first time: select everything
        st.session_state[selected_key] = list(available)
        return

    # prune removed
    current = [n for n in current if n in available]

    # add new
    for n in available:
        if n in newly_added and n not in current:
            current.append(n)

    st.session_state[selected_key] = current


def _ingest_uploads(uploaded_files, *, target: str):
    """Read uploaded files and persist them in session_state.

    target: 'MMS' or 'conv' or 'studies'

    Notes:
      - Every upload is also stored in a global library, so you can later copy/move
        the same file into another mode without re-uploading.
    """
    if uploaded_files is None:
        return

    key = _mode_cache_key(target)

    for f in uploaded_files:
        try:
            bs = f.getvalue()
        except Exception:
            # Streamlit UploadedFile always supports getvalue, but keep it defensive.
            bs = f.read()

        cu = _library_add(f.name, bs)
        if cu.content_hash not in st.session_state[key]:
            st.session_state[key][cu.content_hash] = cu


def _ensure_upload_cache():
    """Initialize session-state caches used to persist uploads across reruns."""
    st.session_state.setdefault("uploads_library", {})  # hash -> _CachedUpload
    st.session_state.setdefault("uploads_library_display", {})  # hash -> display name

    st.session_state.setdefault("uploads_mms", {})  # hash -> _CachedUpload
    st.session_state.setdefault("uploads_conv", {})  # hash -> _CachedUpload
    st.session_state.setdefault("uploads_studies", {})  # hash -> _CachedUpload

    st.session_state.setdefault("aliases_conv", {})  # filename -> alias
    st.session_state.setdefault("mms_selected_files", None)  # list[str] | None
    st.session_state.setdefault("studies_selected_files", None)  # list[str] | None


def _render_move_copy_panel(current_mode: str):
    """Sidebar UI to copy/move already-uploaded files between modes.

    current_mode: one of 'MMS' | 'Convergence' | 'Studies'

    This works on the in-session caches, so it doesn't depend on the live uploader content.
    """
    # Ensure caches are consistent.
    for m in ("MMS", "Convergence", "Studies"):
        _sync_mode_cache_from_library(m)

    cur_key = _mode_cache_key(current_mode)
    with st.sidebar.expander("Move / copy between views", expanded=False):
        st.caption(
            "Accidentally uploaded a dissipation CSV in the wrong view? Use this to copy or move cached uploads "
            "between MMS / Convergence / Studies without re-uploading."
        )

        cur_cache = st.session_state.get(cur_key, {})
        cur_hashes = set(cur_cache.keys())

        # Build a label->hash mapping so we can show duplicates safely.
        labels = {}
        for h, cu in st.session_state.get("uploads_library", {}).items():
            display = st.session_state.get("uploads_library_display", {}).get(h, cu.name)
            # include suffix to disambiguate duplicate filenames
            label = f"{display}  {h[:8]}"
            labels[label] = h

        if not labels:
            st.caption("No cached uploads yet.")
            return

        # Choose source and destination.
        modes = ["MMS", "Convergence", "Studies"]
        dst_mode = current_mode
        src_mode = st.selectbox("Source view", options=[m for m in modes if m != dst_mode], index=0,
                                key=f"mv_src__{dst_mode}")

        src_cache = st.session_state.get(_mode_cache_key(src_mode), {})
        src_hashes = list(src_cache.keys())

        if not src_hashes:
            st.caption(f"No cached uploads in {src_mode}.")
            return

        # Only allow selecting hashes present in the source.
        src_options = []
        for h in src_hashes:
            cu = st.session_state["uploads_library"].get(h)
            if cu is None:
                continue
            display = st.session_state.get("uploads_library_display", {}).get(h, cu.name)
            src_options.append(f"{display}  {h[:8]}")

        selected_labels = st.multiselect(
            "Select file(s)",
            options=sorted(src_options),
            default=[],
            key=f"mv_pick__{dst_mode}__from__{src_mode}",
        )
        picked = [labels[lbl] for lbl in selected_labels if lbl in labels]

        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Copy to {dst_mode}", key=f"copy__{dst_mode}__from__{src_mode}", disabled=not picked):
                _copy_hashes_to_mode(picked, mode=dst_mode)
                st.success(f"Copied {len(picked)} file(s) to {dst_mode}.")
                st.rerun()
        with col2:
            if st.button(f"Move to {dst_mode}", key=f"move__{dst_mode}__from__{src_mode}", disabled=not picked):
                _move_hashes(picked, src_mode=src_mode, dst_mode=dst_mode)
                st.success(f"Moved {len(picked)} file(s) to {dst_mode}.")
                st.rerun()

        st.markdown("---")
        st.caption("Advanced")
        if st.button("Delete selected from library (removes from all views)", key=f"del_lib__{dst_mode}",
                     disabled=not picked):
            for h in picked:
                st.session_state.get("uploads_library", {}).pop(h, None)
                st.session_state.get("uploads_library_display", {}).pop(h, None)
                for m in ("MMS", "Convergence", "Studies"):
                    st.session_state.get(_mode_cache_key(m), {}).pop(h, None)
            st.success(f"Deleted {len(picked)} file(s) from library.")
            st.rerun()


# ------------------------
# Shared utilities
# ------------------------


def _normalize_column_name(c: str) -> str:
    if c is None:
        return ""
    c = str(c)
    c = c.strip().strip('"').strip("'").strip()
    c = re.sub(r"\s+", " ", c)
    return c


def _drop_spacer_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    for c in df.columns:
        if c in {"step", "n"}:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
            continue
        if c in {"Time", "t"} or any(
                str(c).startswith(p)
                for p in (
                        "error_",
                        "delta_",
                        "rel_delta_",
                        "cum_mean_",
                        "cum_rms_",
                        "h",
                        "dt",
                        "E",
                        "a",
                        "adot",
                )
        ):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_canonical_schema(df: pd.DataFrame) -> pd.DataFrame:
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

    # Normalize time column naming.
    # Solver may emit `t` (studies) or `time` (error history); viewer uses `Time` in MMS mode.
    if "Time" not in df.columns:
        if "t" in df.columns:
            df = df.rename(columns={"t": "Time"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "Time"})

    return df


def _available_metrics(df: pd.DataFrame) -> dict:
    families = {
        "Instantaneous": ["error_u_L2", "error_u_H1", "error_v_L2", "error_v_H1"],
        "Cumulative": [
            "cum_mean_u_L2",
            "cum_rms_u_L2",
            "cum_mean_u_H1",
            "cum_rms_u_H1",
            "cum_mean_v_L2",
            "cum_rms_v_L2",
        ],
        "Delta": ["delta_u_L2", "delta_u_H1", "delta_v_L2", "delta_v_H1"],
        "Rel delta": ["rel_delta_u_L2", "rel_delta_u_H1", "rel_delta_v_L2", "rel_delta_v_H1"],
    }
    out = {}
    cols = set(df.columns)
    for fam, keys in families.items():
        out[fam] = any(k in cols for k in keys)
    return out


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


def _default_palette(name: str):
    palettes = {
        "Default": px.colors.qualitative.Plotly,
        "High contrast": ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"],
        "Colorblind-friendly": px.colors.qualitative.Set1,
    }
    return palettes.get(name, px.colors.qualitative.Plotly)


# ------------------------
# Convergence mode
# ------------------------
def _infer_study_schema(df: pd.DataFrame) -> str:
    """Infer which solver study a CSV corresponds to."""
    low = {str(c).strip().lower() for c in df.columns}
    if {"n", "t", "e"}.issubset(low):
        return "Dissipation"
    if {"n", "t", "a", "adot"}.issubset(low):
        return "Modal"
    return "Unknown"


def _render_what_can_i_plot_panel(*, datasets: dict[str, pd.DataFrame], context: str):
    """Render a self-documenting panel describing what the uploaded files enable."""
    with st.sidebar.expander("What can I plot?", expanded=True):
        st.markdown(
            "This app auto-detects what your CSV contains from its column names. "
            "Use this panel to see which plots apply and which **Mode** to use."
        )
        st.caption(f"Context: {context}")

        if not datasets:
            st.caption("No CSVs loaded yet.")
            return

        for name, df in datasets.items():
            cols = list(df.columns)
            st.markdown(f"**{name}**")
            st.caption(f"Columns: {', '.join(cols[:12])}{' ...' if len(cols) > 12 else ''}")

            # Study detection first
            study_kind = _infer_study_schema(df)
            if study_kind in {"Dissipation", "Modal"}:
                st.markdown(f"- Detected: **{study_kind}** study → use **Mode = Studies**")
                if study_kind == "Dissipation":
                    st.markdown("  - Plots: `E(t)` and (if present) `E/E0(t)`")
                else:
                    st.markdown("  - Plots: `a(t)` and `adot(t)`")
                st.markdown("---")
                continue

            # Convergence detection (do this before MMS, otherwise generic `step/time` heuristics can mislead)
            low = {str(c).strip().lower() for c in df.columns}
            if "h" in low or "dt" in low:
                st.markdown("- Detected: **Convergence table** → use **Mode = Convergence**")
                st.markdown("---")
                continue

            # MMS detection: require at least one MMS-specific metric family; time/step alone is too generic.
            mms_avail = _available_metrics(df)
            mms_fams = [k for k, v in mms_avail.items() if v]
            has_time = ("t" in low) or ("time" in low)
            has_step = "step" in low

            if mms_fams:
                st.markdown("- Detected: **MMS / error history** → use **Mode = MMS**")
                st.markdown(
                    f"  - X axis available: {', '.join([x for x in ['Time' if has_time else '', 'step' if has_step else ''] if x]) or '(missing time/step)'}"
                )
                st.markdown(
                    f"  - Metric families available: {', '.join(mms_fams) if mms_fams else '(none detected)'}"
                )
                st.markdown("---")
                continue

            st.markdown("- Detected: **Unknown** (not a recognized solver CSV schema yet)")
            st.markdown("---")


def _ensure_multiselect_state(widget_key: str, *, options: list[str], desired_default: list[str]):
    """Make Streamlit multiselect state robust when options change.

    Streamlit can keep old selections in session_state even if the uploader removed a file.
    This helper ensures the widget state is always a subset of current options; if not, it resets it.
    """
    desired_default = [x for x in desired_default if x in options]
    cur = st.session_state.get(widget_key)
    if cur is None:
        return
    if not isinstance(cur, list):
        st.session_state[widget_key] = desired_default
        return
    if any(x not in options for x in cur):
        st.session_state[widget_key] = desired_default
        return


def render_convergence():
    st.write(
        "Upload convergence CSVs produced by the solver (space/time). "
        "You'll get log-log plots, fitted observed orders, and per-step order tables."
    )

    _render_move_copy_panel("Convergence")

    uploaded_conv = st.file_uploader(
        "Convergence CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="uploader_conv",
        help="Space: h,u_L2,u_H1,v_L2,p_*.  Time: dt,u_L2,u_H1,v_L2,q_*.",
    )

    # If uploader empty, keep cached files (so Move/Copy works) and just show guidance.
    if uploaded_conv:
        _ingest_uploads(uploaded_conv, target="conv")

        # If the user explicitly removed files from the uploader, reflect that in this mode's cache.
        # (We only do this when uploader is non-empty; Streamlit may reset uploaders when switching tabs.)
        keep_names = {f.name for f in uploaded_conv}
        _prune_cache_by_filenames("uploads_conv", keep_names)
    else:
        if not st.session_state.get("uploads_conv"):
            st.info("Upload at least one convergence CSV to start, or use the Move/Copy panel to bring one here.")
            st.stop()

    if not st.session_state["uploads_conv"]:
        st.info("Upload at least one convergence CSV to start.")
        st.stop()

    conv_names = [cu.name for cu in st.session_state["uploads_conv"].values()]
    for n in conv_names:
        st.session_state["aliases_conv"].setdefault(n, n)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Convergence file aliases")
    for n in sorted(conv_names):
        key = f"alias_conv__{n}"
        st.session_state["aliases_conv"][n] = st.sidebar.text_input(
            f"Alias for {n}",
            value=st.session_state["aliases_conv"].get(n, n),
            key=key,
        )

    conv_dfs = []
    for cu in st.session_state["uploads_conv"].values():
        try:
            alias = st.session_state["aliases_conv"].get(cu.name, cu.name)
            df = load_convergence_csv(io.BytesIO(cu.data), name=alias)
            kind = detect_kind(df)
            df = compute_observed_orders(df)
            df["__kind"] = kind.value
            conv_dfs.append(df)
        except Exception as e:
            st.error(f"Failed loading {cu.name}: {e}")

    if not conv_dfs:
        st.stop()

    # Guidance panel based on the parsed convergence tables
    _render_what_can_i_plot_panel(
        datasets={df.get("__name", pd.Series(["run"])).iloc[0]: df for df in conv_dfs},
        context="Convergence mode",
    )

    kinds = sorted({df["__kind"].iloc[0] for df in conv_dfs})
    selected_kind = st.sidebar.selectbox(
        "Convergence kind",
        kinds,
        index=0,
        help="Space = error vs h, Time = error vs dt",
    )

    active = [df for df in conv_dfs if df["__kind"].iloc[0] == selected_kind]
    xcol = "h" if selected_kind == "Space" else "dt"

    st.sidebar.markdown("---")
    metrics = ["u_L2", "u_H1", "v_L2"]
    selected_metrics = st.sidebar.multiselect("Metrics", metrics, default=metrics)

    st.subheader("Error vs resolution (log-log)")
    fig = go.Figure()
    for df in active:
        run_name = df.get("__name", pd.Series(["run"])).iloc[0]
        for m in selected_metrics:
            if m not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=df[xcol],
                    y=df[m],
                    mode="lines+markers",
                    name=f"{run_name} · {pretty_metric_name(m)}",
                )
            )

    fig.update_xaxes(title_text=xcol, type="log")
    fig.update_yaxes(title_text="error", type="log")
    fig.update_layout(
        title=f"{selected_kind.title()} convergence: error vs {xcol}",
        legend=dict(orientation="h"),
        height=520,
        margin=dict(l=40, r=20, t=60, b=50),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Fitted observed order (global fit)")
    fit_tables = []
    for df in active:
        run_name = df.get("__name", pd.Series(["run"])).iloc[0]
        fits = summarize_fits(df, error_cols=selected_metrics)
        if fits.empty:
            continue
        fits.insert(0, "run", run_name)
        fits["metric"] = fits["metric"].map(pretty_metric_name)
        fit_tables.append(fits)

    if fit_tables:
        st.dataframe(pd.concat(fit_tables, ignore_index=True), width="stretch", hide_index=True)

    st.markdown("### Raw data (with per-step orders)")
    for df in active:
        run_name = df.get("__name", pd.Series(["run"])).iloc[0]
        with st.expander(f"{run_name} ({xcol} rows: {len(df)})", expanded=False):
            order_cols = (
                ["p_uL2", "p_uH1", "p_vL2"] if selected_kind == "Space" else ["q_uL2", "q_uH1", "q_vL2"]
            )
            cols = [xcol] + selected_metrics + [c for c in order_cols if c in df.columns]
            if "mesh" in df.columns:
                cols.append("mesh")
            st.dataframe(df[cols], width="stretch", hide_index=True)


# ------------------------
# Studies mode
# ------------------------
def render_studies():
    st.write(
        "Upload CSVs produced by solver studies. The viewer auto-detects each file type by its columns and plots the "
        "relevant quantities."
    )

    _render_move_copy_panel("Studies")

    uploaded_studies = st.file_uploader(
        "Study CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="uploader_studies",
        help=(
            "Dissipation: columns `n,t,E,E_over_E0`.  Modal: columns `n,t,a,adot`."
        ),
    )

    # If uploader is empty, don't wipe caches: user might want to Move/Copy files into Studies.
    if uploaded_studies:
        current_names = set([f.name for f in uploaded_studies])
        prev_names = set(st.session_state.get("_prev_studies_names", set()))
        st.session_state["_prev_studies_names"] = current_names
        newly_added = current_names - prev_names

        # NOTE: don't prune cached uploads based on the current uploader widget state.
        # Streamlit resets the uploader when switching modes/tabs; pruning here would orphan cached files
        # (they'd still exist in the library, but disappear from this view's list and couldn't be removed).

        st.session_state.setdefault("uploads_studies", {})
        _ingest_uploads(uploaded_studies, target="studies")

        # If the user explicitly removed files from the uploader, reflect that in this mode's cache.
        # (We only do this when uploader is non-empty; Streamlit may reset uploaders when switching tabs.)
        _prune_cache_by_filenames("uploads_studies", current_names)
    else:
        newly_added = set()
        if not st.session_state.get("uploads_studies"):
            st.info("Upload one or more study CSV files to get started, or Move/Copy existing uploads into Studies.")
            st.stop()

    if not st.session_state["uploads_studies"]:
        st.info("Upload one or more study CSV files to get started.")
        st.stop()

    # Always build the selectable list from cached uploads, even if parsing fails.
    study_datasets: dict[str, pd.DataFrame] = {}
    study_kinds: dict[str, str] = {}
    errors = []
    all_cached_names: list[str] = []
    for cu in st.session_state["uploads_studies"].values():
        all_cached_names.append(cu.name)
        try:
            df = read_csv_robust(cu.data)
            df = _drop_spacer_columns(df)
            # normalize common columns
            if "t" not in df.columns and "Time" in df.columns:
                df = df.rename(columns={"Time": "t"})
            df = _coerce_numeric_columns(df)
            kind = _infer_study_schema(df)
            study_datasets[cu.name] = df
            study_kinds[cu.name] = kind
        except Exception as e:
            # Keep the file in the list, but mark it as Unknown with an error.
            study_kinds[cu.name] = "Unknown"
            errors.append((cu.name, str(e)))

    if errors:
        with st.sidebar.expander("Files with parsing issues", expanded=False):
            for name, err in errors:
                st.warning(f"{name}: {err}")

    # Provide a palette for study plots
    palette_choice = st.sidebar.selectbox(
        "Color palette", options=["Default", "High contrast", "Colorblind-friendly"], index=0, key="palette_studies"
    )
    palette = _default_palette(palette_choice)

    # Guidance panel: use only successfully parsed datasets
    _render_what_can_i_plot_panel(datasets=study_datasets, context="Studies mode")

    # sidebar selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Study files")
    names = sorted(all_cached_names)

    _autoselect_new_files("studies_selected_files", names, newly_added)

    _ensure_multiselect_state(
        "studies_selected_widget",
        options=names,
        desired_default=st.session_state.get("studies_selected_files") or names,
    )

    selected = st.sidebar.multiselect(
        "Select files",
        options=names,
        default=st.session_state.get("studies_selected_files") or names[:1],
        key="studies_selected_widget",
    )
    selected = [s for s in selected if s in names]
    st.session_state["studies_selected_files"] = selected

    if not selected:
        st.info("No study file selected. Pick one in the sidebar, or Move/Copy a CSV into Studies.")
        st.stop()

    # Only plot files we could parse; still show tabs for all selected.
    tabs = st.tabs([f"{n} ({study_kinds.get(n, 'Unknown')})" for n in selected])
    for n, tab in zip(selected, tabs):
        with tab:
            st.subheader(n)
            st.caption(f"Detected kind: {study_kinds.get(n, 'Unknown')}")
            if n in study_datasets:
                st.dataframe(study_datasets[n].head(30), width="stretch")
            else:
                st.warning(
                    "This CSV couldn't be parsed, so plots are unavailable. You can still Move/Copy it to another view.")

    st.markdown("---")

    by_kind: dict[str, list[str]] = {"Dissipation": [], "Modal": [], "Unknown": []}
    for n in selected:
        by_kind.setdefault(study_kinds.get(n, "Unknown"), []).append(n)

    if by_kind.get("Dissipation"):
        st.subheader("Dissipation study")
        st.caption("Plots from columns: `t`, `E`, and optionally `E_over_E0`.")

        figE = go.Figure()
        figR = go.Figure()
        for i, n in enumerate(by_kind["Dissipation"]):
            df = study_datasets[n]
            cols = {c.lower(): c for c in df.columns}
            tcol = cols.get("t")
            ecol = cols.get("e")
            rcol = cols.get("e_over_e0") or cols.get("e_over_e_0")
            if tcol is None or ecol is None:
                st.warning(f"{n}: missing required columns for dissipation plot (need t and E)")
                continue

            color = palette[i % len(palette)]
            figE.add_trace(
                go.Scatter(
                    x=df[tcol], y=df[ecol], mode="lines+markers", name=f"{n}: E", line=dict(color=color)
                )
            )
            if rcol is not None:
                figR.add_trace(
                    go.Scatter(
                        x=df[tcol],
                        y=df[rcol],
                        mode="lines+markers",
                        name=f"{n}: E/E0",
                        line=dict(color=color),
                    )
                )

        figE.update_xaxes(title_text="t")
        figE.update_yaxes(title_text="Energy E")
        figE.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(figE, width="stretch")

        if len(figR.data) > 0:
            figR.update_xaxes(title_text="t")
            figR.update_yaxes(title_text="E/E0")
            figR.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(figR, width="stretch")

    if by_kind.get("Modal"):
        st.subheader("Modal study")
        st.caption("Plots from columns: `t`, `a`, `adot`.")

        figA = go.Figure()
        figAd = go.Figure()
        for i, n in enumerate(by_kind["Modal"]):
            df = study_datasets[n]
            cols = {c.lower(): c for c in df.columns}
            tcol = cols.get("t")
            acol = cols.get("a")
            adcol = cols.get("adot")
            if tcol is None or acol is None or adcol is None:
                st.warning(f"{n}: missing required columns for modal plot (need t,a,adot)")
                continue
            color = palette[i % len(palette)]
            figA.add_trace(
                go.Scatter(
                    x=df[tcol], y=df[acol], mode="lines+markers", name=f"{n}: a", line=dict(color=color)
                )
            )
            figAd.add_trace(
                go.Scatter(
                    x=df[tcol],
                    y=df[adcol],
                    mode="lines+markers",
                    name=f"{n}: adot",
                    line=dict(color=color),
                )
            )

        figA.update_xaxes(title_text="t")
        figA.update_yaxes(title_text="a(t)")
        figA.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(figA, width="stretch")

        figAd.update_xaxes(title_text="t")
        figAd.update_yaxes(title_text="adot(t)")
        figAd.update_layout(height=420, margin=dict(l=40, r=20, t=40, b=40))
        st.plotly_chart(figAd, width="stretch")

    if by_kind.get("Unknown"):
        st.subheader("Unrecognized study files")
        for n in by_kind["Unknown"]:
            if n not in study_datasets:
                st.warning(
                    f"{n}: could not be parsed, so no plots are available in Studies. You can still Move/Copy it to another view.")
                continue
            df = study_datasets[n]
            st.warning(
                f"{n}: could not detect study type from columns {list(df.columns)}. "
                "Expected dissipation (`n,t,E,E_over_E0`) or modal (`n,t,a,adot`)."
            )


# ------------------------
# MMS mode
# ------------------------
def render_mms():
    st.write(
        "Upload one or more CSVs (with columns like `time`, `error_u_L2`, `error_u_H1`, etc.) and compare plots."
    )

    _render_move_copy_panel("MMS")

    uploaded = st.file_uploader(
        "MMS / error history CSVs",
        type=["csv"],
        accept_multiple_files=True,
        key="uploader_mms",
    )

    # If uploader is empty, don't wipe caches: user might want to Move/Copy files into MMS.
    if uploaded:
        current_names = set([f.name for f in uploaded])
        prev_names = set(st.session_state.get("_prev_mms_names", set()))
        st.session_state["_prev_mms_names"] = current_names
        newly_added = current_names - prev_names

        # NOTE: don't prune cached uploads based on the current uploader widget state.
        # Streamlit resets the uploader when switching modes/tabs; pruning here would orphan cached files.
        _ingest_uploads(uploaded, target="MMS")

        # If the user explicitly removed files from the uploader, reflect that in this mode's cache.
        # (We only do this when uploader is non-empty; Streamlit may reset uploaders when switching tabs.)
        _prune_cache_by_filenames("uploads_mms", current_names)
    else:
        newly_added = set()
        if not st.session_state.get("uploads_mms"):
            st.info("Upload one or more CSV files to get started, or Move/Copy existing uploads into MMS.")
            st.stop()

    if not st.session_state["uploads_mms"]:
        st.info("Upload one or more CSV files to get started.")
        st.stop()

    datasets = {}
    read_errors = []
    all_cached_names: list[str] = []
    for cu in st.session_state["uploads_mms"].values():
        all_cached_names.append(cu.name)
        try:
            df = read_csv_robust(cu.data)
            df = _drop_spacer_columns(df)
            df = _ensure_canonical_schema(df)
            df = _coerce_numeric_columns(df)
            datasets[cu.name] = df
        except Exception as e:
            # Keep the file visible/selectable even if it can't be parsed.
            read_errors.append((cu.name, str(e)))

    _render_what_can_i_plot_panel(datasets=datasets, context="MMS mode")

    if read_errors:
        with st.sidebar.expander("Files with parsing issues", expanded=False):
            for name, err in read_errors:
                st.warning(f"{name}: {err}")

    # IMPORTANT: build the list from cached uploads, not only from successfully parsed datasets.
    # Otherwise schema-mismatched files disappear from the list and can't be removed/moved.
    file_names = sorted({*all_cached_names})

    # Auto-select new files and prune removed from selection.
    _autoselect_new_files("mms_selected_files", file_names, newly_added)

    # Also prune the stored selection if the user removed files from the uploader (same caveat as above).
    if uploaded:
        st.session_state["mms_selected_files"] = [n for n in (st.session_state.get("mms_selected_files") or []) if n in file_names]

    # Ensure widget state is valid for current options (fixes stale selection when last file is removed)
    _ensure_multiselect_state(
        "mms_selected_widget",
        options=file_names,
        desired_default=st.session_state.get("mms_selected_files") or file_names,
    )

    # sidebar controls
    st.sidebar.header("Files & Plots")
    selected = st.sidebar.multiselect(
        "Select files to plot (overlay)",
        options=file_names,
        default=st.session_state.get("mms_selected_files") or file_names[:1],
        key="mms_selected_widget",
    )
    selected = [s for s in selected if s in file_names]
    st.session_state["mms_selected_files"] = selected

    if not selected:
        st.sidebar.warning("Select at least one file to plot.")
        st.stop()

    # required x-axis (only enforce for files that successfully parsed)
    bad_for_plot = []
    for name in selected:
        if name not in datasets:
            bad_for_plot.append(name)
            continue
        if "Time" not in datasets[name].columns and "step" not in datasets[name].columns:
            bad_for_plot.append(name)

    if bad_for_plot:
        st.error(
            "Some selected files aren't plottable in MMS (missing `time`/`step` or failed parsing): "
            + ", ".join(bad_for_plot)
            + ".\n\nTip: move them to Studies/Convergence using the Move/Copy panel, or deselect them here."
        )
        st.stop()

    # Decide x-axis (time preferred)
    x_axis = st.sidebar.selectbox("X axis", options=["Time", "step"], index=0)
    # If any selected file lacks time, fall back to step automatically.
    if x_axis == "Time" and any("Time" not in datasets[n].columns for n in selected):
        x_axis = "step"
        st.sidebar.info("Some files are missing `time`; using `step` on the x-axis.")

    # sidebar plot toggles
    show_inst = st.sidebar.checkbox("Instantaneous errors (log scale)", value=True)
    show_delta = st.sidebar.checkbox("Step-to-step deltas", value=False)
    show_reldelta = st.sidebar.checkbox("Relative step-to-step deltas", value=False)
    show_table = st.sidebar.checkbox("Show comparison table", value=False)
    legend_mode = st.sidebar.selectbox("Legend placement", options=["Outside (compact)", "Inside (default)"], index=0)

    # Theming & palettes (Light theme removed)
    palette_choice = st.sidebar.selectbox(
        "Color palette", options=["Default", "High contrast", "Colorblind-friendly"], index=0
    )

    palettes = {
        "Default": px.colors.qualitative.Plotly,
        "High contrast": ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"],
        "Colorblind-friendly": px.colors.qualitative.Set1,
    }
    palette = palettes.get(palette_choice, px.colors.qualitative.Plotly)

    if "aliases" not in st.session_state:
        st.session_state.aliases = {name: name for name in file_names}

    st.sidebar.markdown("---")
    st.sidebar.subheader("File aliases")
    for name in file_names:
        key = f"alias__{name}"
        val = st.sidebar.text_input(f"Alias for {name}", value=st.session_state.aliases.get(name, name), key=key)
        st.session_state.aliases[name] = val

    st.sidebar.markdown("---")
    st.sidebar.subheader("Metric selection")

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

    # ------------------------
    # Plotting helpers
    # ------------------------
    def _get_x(df: pd.DataFrame):
        if x_axis in df.columns:
            return df[x_axis]
        if "Time" in df.columns:
            return df["Time"]
        if "step" in df.columns:
            return df["step"]
        return df.index

    def _col(metric: str, var: str, norm: str):
        return f"{metric}_{var}_{norm}"

    # ------------------------
    # MMS plots
    # ------------------------
    if show_inst:
        st.subheader("Instantaneous MMS error (interactive)")
        fig = go.Figure()

        any_trace = False
        for i, name in enumerate(selected):
            df = datasets[name]
            x = _get_x(df)
            alias = st.session_state.aliases.get(name, name)
            color = palette[i % len(palette)]

            for var, label in (("u", "u"), ("v", "v")):
                c = _col("error", var, metric_norm)
                if c not in df.columns:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[c],
                        mode="lines+markers",
                        name=f"{alias}: {label} ({metric_norm})",
                        line=dict(color=color, width=2),
                    )
                )
                any_trace = True

        if not any_trace:
            st.warning(
                f"No instantaneous error columns found for norm={metric_norm} (expected error_u_{metric_norm}/error_v_{metric_norm}).")
        else:
            fig.update_xaxes(title_text=x_axis)
            fig.update_yaxes(title_text=f"{metric_norm} error", type="log")
            fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

    if show_delta:
        st.subheader("Step-to-step deltas (interactive)")
        fig = go.Figure()
        any_trace = False

        for i, name in enumerate(selected):
            df = datasets[name]
            x = _get_x(df)
            alias = st.session_state.aliases.get(name, name)
            color = palette[i % len(palette)]

            for var, label in (("u", "u"), ("v", "v")):
                c = _col("delta", var, metric_norm)
                if c not in df.columns:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[c],
                        mode="lines+markers",
                        name=f"{alias}: Δ{label} ({metric_norm})",
                        line=dict(color=color, width=2),
                    )
                )
                any_trace = True

        if not any_trace:
            st.warning(
                f"No delta columns found for norm={metric_norm} (expected delta_u_{metric_norm}/delta_v_{metric_norm}).")
        else:
            fig.update_xaxes(title_text=x_axis)
            fig.update_yaxes(title_text="Δ error")
            fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

    if show_reldelta:
        st.subheader("Relative step-to-step deltas (interactive)")
        fig = go.Figure()
        any_trace = False

        for i, name in enumerate(selected):
            df = datasets[name]
            x = _get_x(df)
            alias = st.session_state.aliases.get(name, name)
            color = palette[i % len(palette)]

            for var, label in (("u", "u"), ("v", "v")):
                c = _col("rel_delta", var, metric_norm)
                if c not in df.columns:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=df[c],
                        mode="lines+markers",
                        name=f"{alias}: relΔ{label} ({metric_norm})",
                        line=dict(color=color, width=2),
                    )
                )
                any_trace = True

        if not any_trace:
            st.warning(
                f"No relative-delta columns found for norm={metric_norm} (expected rel_delta_u_{metric_norm}/rel_delta_v_{metric_norm})."
            )
        else:
            fig.update_xaxes(title_text=x_axis)
            fig.update_yaxes(title_text="relative Δ error")
            fig.update_layout(height=520, margin=dict(l=40, r=20, t=40, b=40), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

    # Note: show_table was present in the previous iteration; for now, keep it as a placeholder.
    if show_table:
        st.info(
            "Comparison table refactor: not yet reintroduced. If you want it back, I'll add the summary table + preset save/load next.")


# ------------------------
# Main router
# ------------------------

def _render_global_sidebar():
    with st.sidebar.expander("Uploads (persisted)", expanded=False):
        st.caption("Uploads are cached in-session so switching mode won't clear them.")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Clear MMS uploads"):
                st.session_state["uploads_mms"] = {}
                st.session_state["mms_selected_files"] = None
                st.rerun()
        with col_b:
            if st.button("Clear Conv uploads"):
                st.session_state["uploads_conv"] = {}
                st.rerun()
        with col_c:
            if st.button("Clear Studies uploads"):
                st.session_state["uploads_studies"] = {}
                st.session_state["studies_selected_files"] = None
                st.rerun()

        st.write(f"MMS cached: {len(st.session_state.get('uploads_mms', {}))}")
        st.write(f"Conv cached: {len(st.session_state.get('uploads_conv', {}))}")
        st.write(f"Studies cached: {len(st.session_state.get('uploads_studies', {}))}")


# init persistence
_ensure_upload_cache()

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Viewer",
    options=["MMS", "Convergence", "Studies"],
    index=0,
    help="Switch between MMS time-series error viewer, convergence study viewer, and solver study CSV viewers.",
)

_render_global_sidebar()

if mode == "MMS":
    render_mms()
elif mode == "Convergence":
    render_convergence()
else:
    render_studies()

st.stop()
