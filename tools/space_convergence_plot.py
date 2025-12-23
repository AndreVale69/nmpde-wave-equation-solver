import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "h" not in df.columns:
        raise ValueError(f"Missing required column 'h' in {csv_path}")
    # Sort by h descending so the plot looks natural (large h on the left once inverted)
    df = df.sort_values("h", ascending=False).reset_index(drop=True)
    return df


def _finite_first(y: np.ndarray) -> float:
    m = np.isfinite(y)
    if not np.any(m):
        return float("nan")
    return float(y[m][0])


def plot_comparison(
    *,
    be_csv: Path,
    cn_csv: Path,
    dt_label: str,
    out: Path | None,
    show: bool,
    strict: bool,
) -> None:
    datasets: list[tuple[str, Path]] = [
        ("Backward Euler", be_csv),
        ("Crank–Nicolson", cn_csv),
    ]

    loaded: list[tuple[str, pd.DataFrame]] = []
    for name, path in datasets:
        try:
            loaded.append((name, _load_csv(path)))
        except Exception as e:
            if strict:
                raise
            print(f"[warn] Skipping {name}: {e}")

    if not loaded:
        raise SystemExit("No datasets loaded. Check CSV paths.")

    # Norms we try to plot; attach pretty labels + expected reference order.
    norms: list[tuple[str, str, float]] = [
        ("u_L2", r"$\|u\|_{L^2}$", 2.0),
        ("u_H1", r"$\|u\|_{H^1}$", 1.0),
        ("v_L2", r"$\|v\|_{L^2}$", 2.0),
    ]

    # Keep colors consistent per norm, but distinguish methods by marker/linestyle.
    norm_colors = {
        "u_L2": "C0",
        "u_H1": "C1",
        "v_L2": "C2",
    }
    method_style = {
        "Backward Euler": dict(marker="o", linestyle="-"),
        "Crank–Nicolson": dict(marker="s", linestyle="--"),
    }

    plt.figure(figsize=(7, 5))

    # Plot datasets
    for method_name, df in loaded:
        h = df["h"].to_numpy(dtype=float)

        for col, pretty, order in norms:
            if col not in df.columns:
                print(f"[warn] {method_name}: missing column '{col}', skipping")
                continue

            y = df[col].to_numpy(dtype=float)
            style = method_style.get(method_name, {})
            color = norm_colors.get(col, None)
            label = f"{pretty} ({method_name})"
            plt.loglog(h, y, label=label, color=color, **style)

            # Reference slope guide scaled to the first finite point of this curve.
            y0 = _finite_first(y)
            h0 = h[0] if len(h) else float("nan")
            if np.isfinite(y0) and np.isfinite(h0) and h0 > 0:
                C = y0 / (h0**order)
                h_ref = np.array([h[0], h[-1]], dtype=float)
                # Only add one legend entry per order (avoid duplicates)
                slope_label = None
                if order == 2.0:
                    slope_label = r"$\mathcal{O}(h^2)$"
                elif order == 1.0:
                    slope_label = r"$\mathcal{O}(h)$"
                plt.loglog(
                    h_ref,
                    C * h_ref**order,
                    color="k",
                    linestyle=":" if order == 2.0 else "-.",
                    alpha=0.35,
                    linewidth=1.2,
                    label=slope_label,
                )

    # Deduplicate legend entries (matplotlib will repeat slope labels otherwise)
    handles, labels = plt.gca().get_legend_handles_labels()
    seen: set[str] = set()
    dedup_h = []
    dedup_l = []
    for hnd, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        dedup_h.append(hnd)
        dedup_l.append(lab)

    plt.gca().invert_xaxis()
    plt.xlabel(r"$h$")
    plt.ylabel("Error norm")
    plt.title(f"Spatial convergence comparison ($\\Delta t = {dt_label}$)")
    plt.grid(True, which="both")
    plt.legend(dedup_h, dedup_l, fontsize=9)
    plt.tight_layout()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"[info] Wrote figure: {out}")

    if show:
        plt.show()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(
        description="Compare spatial convergence between Backward Euler and Crank–Nicolson."
    )
    p.add_argument(
        "--be-csv",
        type=Path,
        default=repo_root
        / "results/space_convergence/backward_euler/space_convergence-backward_euler-dt0_01.csv",
        help="Path to Backward Euler space convergence CSV",
    )
    p.add_argument(
        "--cn-csv",
        type=Path,
        default=repo_root
        / "results/space_convergence/crank_nicolson/space_convergence-crank_nicolson-dt0_01.csv",
        help="Path to Crank–Nicolson space convergence CSV",
    )
    p.add_argument(
        "--dt-label",
        default="0.01",
        help="Label for time step shown in the title (e.g., 0.01)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path (e.g., analysis/figures/space_conv_compare.png)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window (useful in headless runs)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any dataset can't be loaded",
    )

    args = p.parse_args()

    plot_comparison(
        be_csv=args.be_csv,
        cn_csv=args.cn_csv,
        dt_label=args.dt_label,
        out=args.out,
        show=not args.no_show,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()

