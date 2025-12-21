import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MMS Plot Viewer", layout="wide")

st.title("MMS Plot Viewer")
st.write("Upload a CSV (with columns like `time`, `error_u`, `error_v`, etc.) and view plots.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head(20), width='stretch')

required_cols = {"time"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain a `time` column.")
    st.stop()

t = df["time"]

# Sidebar controls
st.sidebar.header("Plots")
show_inst = st.sidebar.checkbox("Instantaneous errors (log scale)", value=True)
show_cum = st.sidebar.checkbox("Cumulative statistics", value=True)
show_delta = st.sidebar.checkbox("Step-to-step deltas", value=False)

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

# ---------- FIG 1: instantaneous errors ----------
if show_inst:
    st.subheader("Instantaneous MMS error")
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    if "error_u" in df.columns:
        ax.semilogy(t, df["error_u"], "o-", label=r"$\|u-u_{ex}\|_{L^2}$")
    else:
        st.warning("Missing column: error_u")

    if "error_v" in df.columns:
        ax.semilogy(t, df["error_v"], "s--", label=r"$\|v-v_{ex}\|_{L^2}$")
    else:
        st.warning("Missing column: error_v")

    ax.set_xlabel("Time")
    ax.set_ylabel("L2 error")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    st.pyplot(fig)

    st.download_button(
        "Download: instantaneous_errors.png",
        data=fig_to_png_bytes(fig),
        file_name="instantaneous_errors.png",
        mime="image/png",
    )
    st.download_button(
        "Download: instantaneous_errors.pdf",
        data=fig_to_pdf_bytes(fig),
        file_name="instantaneous_errors.pdf",
        mime="application/pdf",
    )


# ---------- FIG 2: cumulative stats ----------
if show_cum:
    st.subheader("Cumulative error statistics")
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    # u
    if "cum_mean_u" in df.columns:
        ax.plot(t, df["cum_mean_u"], label="Mean(u)")
    if "cum_rms_u" in df.columns:
        ax.plot(t, df["cum_rms_u"], "--", label="RMS(u)")

    # v
    if "cum_mean_v" in df.columns:
        ax.plot(t, df["cum_mean_v"], label="Mean(v)")
    if "cum_rms_v" in df.columns:
        ax.plot(t, df["cum_rms_v"], "--", label="RMS(v)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Error magnitude")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.download_button(
        "Download: cumulative_stats.png",
        data=fig_to_png_bytes(fig),
        file_name="cumulative_stats.png",
        mime="image/png",
    )
    st.download_button(
        "Download: cumulative_stats.pdf",
        data=fig_to_pdf_bytes(fig),
        file_name="cumulative_stats.pdf",
        mime="application/pdf",
    )

# ---------- FIG 3: deltas ----------
if show_delta:
    st.subheader("Step-to-step variation")
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    if "delta_u" in df.columns:
        ax.plot(t, df["delta_u"], label="delta_u")
    else:
        st.warning("Missing column: delta_u")

    if "delta_v" in df.columns:
        ax.plot(t, df["delta_v"], label="delta_v")
    else:
        st.warning("Missing column: delta_v")

    ax.set_xlabel("Time")
    ax.set_ylabel("Increment")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.download_button(
        "Download: deltas.png",
        data=fig_to_png_bytes(fig),
        file_name="deltas.png",
        mime="image/png",
    )
    st.download_button(
        "Download: deltas.pdf",
        data=fig_to_pdf_bytes(fig),
        file_name="deltas.pdf",
        mime="application/pdf",
    )
