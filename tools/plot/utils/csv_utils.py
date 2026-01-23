"""CSV utility helpers shared by plotting/analysis tools.

These functions are designed to be robust to common issues when files are produced by
numerical solvers and consumed by web UIs (e.g. Selenium-driven tests):
- UTF-8 BOM
- Windows line endings
- a truncated last line due to a concurrent write / race
- occasional malformed rows

Keep this module dependency-light (pandas only).
"""

from __future__ import annotations

import io

import pandas as pd


def read_csv_robust(data: bytes, *, sep: str = ",") -> pd.DataFrame:
    """Parse CSV bytes robustly.

    Notes
    -----
    - Uses UTF-8 with BOM stripping (utf-8-sig).
    - Drops a potentially truncated last line if the file doesn't end with a newline.
    - Uses the python engine and skips bad lines.
    """

    text = data.decode("utf-8-sig", errors="replace")

    # If the last line is truncated (no trailing newline), drop it.
    if text and not text.endswith("\n") and "\n" in text:
        text = text.rsplit("\n", 1)[0] + "\n"

    # Normalize Windows line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    return pd.read_csv(
        io.StringIO(text),
        engine="python",
        on_bad_lines="skip",
        sep=sep,
    )
