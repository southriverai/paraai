import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel


class ModelPlane(BaseModel):
    model_name: str
    weight_kg: float
    wing_area_m2: float
    drag_coefficient_zero_lift: float
    aspect_ratio: float
    Oswald_efficiency_factor: float


def calculate_drag(model_plane: ModelPlane, velocity: np.ndarray):
    rho = 1.225  # air density (kg/m^3) - sea level
    S = model_plane.wing_area_m2  # wing area (m^2) - typical sailplane
    W = model_plane.weight_kg * 9.81  # weight (N) ~450 kg all-up
    CD0 = model_plane.drag_coefficient_zero_lift  # zero-lift drag coefficient
    AR = model_plane.aspect_ratio  # aspect ratio
    e = model_plane.Oswald_efficiency_factor  # Oswald efficiency factor
    k = 1 / (np.pi * e * AR)
    D_parasite = 0.5 * rho * velocity**2 * S * CD0
    D_induced = (2 * k * W**2) / (rho * V**2 * S)
    D_total = D_parasite + D_induced
    return D_parasite, D_induced, D_total


def calculate_sink(model_plane: ModelPlane, velocity: np.ndarray) -> np.ndarray:
    D_parasite, D_induced, D_total = calculate_drag(model_plane, velocity)
    return D_total * velocity / model_plane.weight_kg * 9.81


def plot_drag(model_plane: ModelPlane, velocity: np.ndarray):
    D_parasite, D_induced, D_total = calculate_drag(model_plane, velocity)
    plt.plot(velocity, D_total, label="Total Drag")
    plt.plot(velocity, D_parasite, "--", label="Parasite Drag")
    plt.plot(velocity, D_induced, "--", label="Induced Drag")
    plt.xlabel("True Airspeed V (m/s)")
    plt.ylabel("Drag Force D (N)")
    plt.title(f"Velocity vs Drag ({model_plane.model_name} Drag Polar)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_sink_rate(model_plane: ModelPlane, velocity: np.ndarray):
    sink_rate = calculate_sink(model_plane, velocity)
    plt.plot(velocity, sink_rate, label="Sink Rate")
    plt.xlabel("True Airspeed V (m/s)")
    plt.ylabel("Sink Rate (m/s)")
    plt.title(f"Velocity vs Sink Rate ({model_plane.model_name} Sink Rate)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------------------------
# Physical / aircraft parameters
# -------------------------

model_plane = ModelPlane(
    model_name="Plane",
    weight_kg=450,  # weight (kg) ~450 kg all-up
    wing_area_m2=15.0,
    drag_coefficient_zero_lift=0.015,
    aspect_ratio=20.0,  # aspect ratio
    Oswald_efficiency_factor=0.85,
)  # Oswald efficiency factor


model_glider = ModelPlane(
    model_name="Glider",
    weight_kg=100,  # weight (kg)
    wing_area_m2=25.0,
    drag_coefficient_zero_lift=0.02,
    aspect_ratio=12.0,  # aspect ratio
    Oswald_efficiency_factor=0.85,
)  # Oswald efficiency factor

# -------------------------
# Velocity range
# -------------------------
V = np.linspace(15, 70, 400)  # m/s (≈ 55–250 km/h)

# -------------------------
# Drag components
# -------------------------
D_parasite, D_induced, D_total = calculate_drag(model_plane, V)
D_parasite_glider, D_induced_glider, D_total_glider = calculate_drag(model_glider, V)
# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(8, 5))
plot_drag(model_plane, V)
plt.show()

plt.figure(figsize=(8, 5))
plot_drag(model_glider, V)
plt.show()


plt.figure(figsize=(8, 5))
plot_sink_rate(model_plane, V)
plt.show()
