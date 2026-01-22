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
