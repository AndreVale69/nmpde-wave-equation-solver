import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("results/space_convergence/backward_euler/dt0_01.csv")

h = df["h"].to_numpy()

u_L2 = df["u_L2"].to_numpy()
u_H1 = df["u_H1"].to_numpy()
v_L2 = df["v_L2"].to_numpy()

plt.figure(figsize=(6,5))

plt.loglog(h, u_L2, "o-", label=r"$\|u\|_{L^2}$")
plt.loglog(h, u_H1, "s-", label=r"$\|u\|_{H^1}$")
plt.loglog(h, v_L2, "d-", label=r"$\|v\|_{L^2}$")

# Reference slopes (scaled to first point)
h_ref = np.array([h[0], h[-1]])

C_uL2 = u_L2[0] / h[0]**2
C_uH1 = u_H1[0] / h[0]
C_vL2 = v_L2[0] / h[0]**2

plt.loglog(h_ref, C_uL2 * h_ref**2, "k--", alpha=0.6, label=r"$\mathcal{O}(h^2)$")
plt.loglog(h_ref, C_uH1 * h_ref, "k-.", alpha=0.6, label=r"$\mathcal{O}(h)$")

plt.gca().invert_xaxis()
plt.xlabel(r"$h$")
plt.ylabel("Error norm")
plt.title("Spatial convergence (Backward Euler, $T=1$)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
