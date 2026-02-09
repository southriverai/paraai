import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Physical / aircraft parameters
# -------------------------
rho = 1.225          # air density (kg/m^3) - sea level
S = 15.0             # wing area (m^2) - typical sailplane
W = 450 * 9.81       # weight (N) ~450 kg all-up
CD0 = 0.015          # zero-lift drag coefficient
AR = 20.0            # aspect ratio
e = 0.85             # Oswald efficiency factor
k = 1 / (np.pi * e * AR)

# -------------------------
# Velocity range
# -------------------------
V = np.linspace(15, 70, 400)  # m/s (≈ 55–250 km/h)

# -------------------------
# Drag components
# -------------------------
D_parasite = 0.5 * rho * V**2 * S * CD0
D_induced = (2 * k * W**2) / (rho * V**2 * S)
D_total = D_parasite + D_induced

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(V, D_total, label="Total Drag")
plt.plot(V, D_parasite, "--", label="Parasite Drag")
plt.plot(V, D_induced, "--", label="Induced Drag")

plt.xlabel("True Airspeed V (m/s)")
plt.ylabel("Drag Force D (N)")
plt.title("Velocity vs Drag (Sailplane Drag Polar)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
