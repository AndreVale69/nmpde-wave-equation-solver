"""Batch convergence plotter.

Usage:
  python tools/plot/plot_convergence.py --csv build/new-space_convergence.csv --out build/space_report.html

Outputs a self-contained HTML file with log-log plots and summary tables.

This intentionally uses Plotly (already a dependency of the Streamlit viewer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from tools.plot.convergence_utils import (
        ConvergenceKind,
        compute_observed_orders,
        detect_kind,
        load_convergence_csv,
        pretty_metric_name,
        summarize_fits,
    )
except Exception:  # pragma: no cover
    from convergence_utils import (
        ConvergenceKind,
        compute_observed_orders,
        detect_kind,
        load_convergence_csv,
        pretty_metric_name,
        summarize_fits,
    )


def build_figure(df: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    kind = detect_kind(df)
    xcol = "h" if kind == ConvergenceKind.SPACE else "dt"

    df = compute_observed_orders(df)

    fig = make_subplots(
        rows=1,
        cols=1,
    )

    for col in ["u_L2", "u_H1", "v_L2"]:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df[xcol],
                y=df[col],
                mode="lines+markers",
                name=pretty_metric_name(col),
            )
        )

    fig.update_xaxes(title_text=xcol, type="log")
    fig.update_yaxes(title_text="error", type="log")
    fig.update_layout(
        title=title or f"{kind.value.title()} convergence",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=50),
    )

    return fig


def build_html_report(df: pd.DataFrame, *, title: str) -> str:
    kind = detect_kind(df)
    xcol = "h" if kind == ConvergenceKind.SPACE else "dt"
    df_with_orders = compute_observed_orders(df)

    fits = summarize_fits(df)
    fits["metric"] = fits["metric"].map(pretty_metric_name)

    fig = build_figure(df, title=title)

    # Basic HTML template; avoids new deps.
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 14px; }}
    h1, h2 {{ margin: 0.2em 0; }}
    .note {{ color: #555; margin-bottom: 14px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="note">Log–log plot of errors vs <code>{xcol}</code>, plus fitted slopes (observed order).</div>

  {plot_div}

  <h2>Fitted slopes</h2>
  {fits_table}

  <h2>Raw data</h2>
  {raw_table}
</body>
</html>
"""

    plot_div = fig.to_html(include_plotlyjs="cdn", full_html=False)
    fits_table = fits.to_html(index=False, float_format=lambda x: f"{x:.4g}")
    raw_table = df_with_orders.to_html(index=False, float_format=lambda x: f"{x:.6g}")

    return html.format(
        title=title,
        xcol=xcol,
        plot_div=plot_div,
        fits_table=fits_table,
        raw_table=raw_table,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="Convergence CSV file path")
    ap.add_argument(
        "--out",
        required=True,
        help="Output HTML file path (report).",
    )
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    # For now, report supports one CSV. Accept multiple to extend later.
    if len(args.csv) != 1:
        raise SystemExit("Currently supports a single --csv. Provide exactly one.")

    csv_path = Path(args.csv[0])
    df = load_convergence_csv(csv_path, name=csv_path.stem)

    title = args.title or f"{csv_path.name}"
    html = build_html_report(df, title=title)

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
