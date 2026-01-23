import json

import pandas as pd


def df_summary_for_llm(df: pd.DataFrame, max_head_rows: int = 8) -> dict:
    """Generate a summary of a DataFrame for LLM consumption.
    Args:
        df (pd.DataFrame): The DataFrame to summarize.
        max_head_rows (int): Maximum number of rows from the head to include.
    Returns:
        dict: A summary dictionary containing shape, columns, head, and numeric stats.
    """
    numeric = df.select_dtypes(include="number")
    summary = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "head": df.head(max_head_rows).to_dict(orient="records"),
    }

    if not numeric.empty:
        summary["numeric_stats"] = {
            "min": numeric.min(numeric_only=True).to_dict(),
            "max": numeric.max(numeric_only=True).to_dict(),
            "mean": numeric.mean(numeric_only=True).to_dict(),
            "last_row": numeric.tail(1).to_dict(orient="records")[0],
        }

    return summary


def build_interpretation_prompt(kind: str, df: pd.DataFrame) -> str:
    summary = df_summary_for_llm(df)

    # Very important: constrain style + structure so it doesn't hallucinate wildly
    return f"""
You are an assistant helping interpret numerical PDE solver results.
Return ONLY Markdown.

Dataset type: {kind}

Your job:
- Explain what the plot/data indicates in plain language (for non-experts)
- Highlight key observations based only on the provided stats and head rows
- Point out any red flags (instability, non-monotone convergence, energy growth, etc.)
- Provide 3 actionable suggestions (e.g., reduce dt, refine mesh, check BCs)

Rules:
- Do NOT invent fields/columns not present.
- If something is ambiguous, say it's ambiguous.
- Keep it concise but informative.

Here is the dataset summary (JSON):
```json
{json.dumps(summary, indent=2)}
````

""".strip()

def build_comparison_prompt(kind: str, df_a: pd.DataFrame, name_a: str, df_b: pd.DataFrame, name_b: str) -> str:
    """
    Build a compact, LLM-friendly prompt to compare two runs of the same analysis kind.
    We do NOT send full CSVs: we send summaries that are stable and small.
    """
    def cols(df):
        return sorted(list(df.columns))

    # Minimal, kind-specific summary
    if kind == "mms":
        metrics = [c for c in df_a.columns if c.startswith("error_")]
        x = "time" if "time" in df_a.columns else ("t" if "t" in df_a.columns else "n")

        def summarize_mms(df):
            m = [c for c in df.columns if c.startswith("error_")]
            peak = df[m].max(numeric_only=True).to_dict() if m else {}
            last = df[[x] + m].dropna().tail(1).to_dict(orient="records")
            return {"x": x, "metrics": m, "peak": peak, "last": last[0] if last else {}}

        sA, sB = summarize_mms(df_a), summarize_mms(df_b)

        return f"""
You are analyzing verification results from the Method of Manufactured Solutions (MMS).
Compare two solver runs A and B. Be precise, avoid speculation, use short bullet points.

Run A: {name_a}
Run B: {name_b}

Columns A: {cols(df_a)}
Columns B: {cols(df_b)}

Summary A: {sA}
Summary B: {sB}

Tasks:
1) Which run is more accurate overall? Use peak and final-time snapshot.
2) Do you see signs of instability or transient spikes? Compare peak vs last.
3) Provide 2-3 actionable suggestions (dt/h refinement, scheme changes), if appropriate.
Return Markdown.
""".strip()

    if kind == "study_dissipation":
        def summarize_energy(df):
            out = {}
            if "e" in df.columns:
                out["E0"] = float(df["e"].iloc[0])
                out["E_end"] = float(df["e"].iloc[-1])
                out["rel_change"] = float((df["e"].iloc[-1] - df["e"].iloc[0]) / df["e"].iloc[0]) if df["e"].iloc[0] != 0 else None
            if "e_over_e0" in df.columns:
                out["min_E_over_E0"] = float(df["e_over_e0"].min())
                out["max_E_over_E0"] = float(df["e_over_e0"].max())
            return out

        return f"""
Compare two energy-dissipation studies (same physical setup, different discretization/scheme).
Run A: {name_a} summary: {summarize_energy(df_a)}
Run B: {name_b} summary: {summarize_energy(df_b)}

Tasks:
1) Which run is more dissipative? (compare E(t) trend and E/E0 extrema)
2) Is there non-physical energy growth in either run?
3) Give a short conclusion (1-2 sentences) and 2 practical suggestions.
Return Markdown.
""".strip()

    if kind == "study_modal":
        def summarize_modal(df):
            out = {}
            if "a" in df.columns:
                out["max_abs_a"] = float(df["a"].abs().max())
                out["final_a"] = float(df["a"].iloc[-1])
            if "adot" in df.columns:
                out["max_abs_adot"] = float(df["adot"].abs().max())
                out["final_adot"] = float(df["adot"].iloc[-1])
            return out

        return f"""
Compare two modal studies (amplitude a(t) and its time-derivative adot(t)).
Run A: {name_a} summary: {summarize_modal(df_a)}
Run B: {name_b} summary: {summarize_modal(df_b)}

Tasks:
1) Compare decay/damping of a(t): which run preserves modal amplitude more?
2) Compare adot(t): any phase/frequency drift indications?
3) Give a short conclusion and 2 suggestions (dt refinement, scheme choice, damping).
Return Markdown.
""".strip()

    if kind == "conv":
        # keep it small: compare fitted slopes if available
        # df already normalized (u_L2/u_H1/v_L2 + h/dt)
        x_col = "dt" if "dt" in df_a.columns else "h"
        y_cols = [c for c in ["u_L2", "u_H1", "v_L2"] if c in df_a.columns]

        return f"""
Compare two convergence studies.
Run A: {name_a} has x={x_col}, errors={y_cols}
Run B: {name_b} has x={x_col}, errors={y_cols}

You must:
1) Compare observed convergence orders: is one closer to expected/theoretical order?
2) Identify if either run is not in the asymptotic regime (irregular slopes).
3) Provide a short conclusion and 2 suggestions.
Return Markdown.
""".strip()

    # fallback
    return f"""
Compare two solver outputs of type '{kind}'.
Run A: {name_a}, columns: {cols(df_a)}
Run B: {name_b}, columns: {cols(df_b)}
Return Markdown with differences and a short conclusion.
""".strip()

