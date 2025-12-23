import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

T = 1.0

files = {
    "theta1":   "results/time_convergence/backward_euler/time_convergence-backward_euler-h0_01.csv",
    "theta0.5": "results/time_convergence/crank_nicolson/time_convergence-crank_nicolson-h0_01.csv",
    "theta0": "results/time_convergence/explicit_euler/time_convergence-explicit_euler-h0_01.csv",
    "theta0.7":   "results/time_convergence/theta07_implicit/time_convergence-theta07_implicit-h0_01.csv",
}

def divides_T(dt, T=1.0, tol=1e-12):
    k = T / dt
    return np.abs(k - np.round(k)) < tol

norms = ["u_L2", "u_H1", "v_L2"]

for norm in norms:
    plt.figure()
    for label, path in files.items():
        df = pd.read_csv(path)

        dt = df["dt"].to_numpy()
        e  = df[norm].to_numpy()

        good = np.array([divides_T(dti, T) for dti in dt])
        bad  = ~good

        # good points (exactly hits T)
        plt.loglog(dt[good], e[good], marker="o", linestyle="-", label=f"{label} (hits T)")

        # bad points (overshoot T -> wrong comparison)
        if bad.any():
            plt.loglog(dt[bad], e[bad], marker="x", linestyle="None", label=f"{label} (overshoots)")

    plt.gca().invert_xaxis()
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(norm)
    plt.title(f"Time convergence comparison: {norm} at t = T")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

plt.show()
