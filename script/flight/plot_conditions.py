from pathlib import Path

from paraai.flight_conditions import FlightConditions
from paraai.tools_plot import plot_flight_conditions


def plot_conditions(show: bool = True):
    path_file = Path("data", "images", "conditions_05.png")
    flight_conditions = FlightConditions(
        take_off_time_s=0,
        take_off_altitude_m=1000,
        landing_time_s=3600 * 2,
        thermal_ceiling_m=1000,
        thermal_net_climb_min_m_s=0.1,
        thermal_net_climb_mean_m_s=0.5,
        thermal_net_climb_std_m_s=0.5,
        thermal_net_climb_max_m_s=8,
        thermal_distance_min_m=1000,
        thermal_distance_mean_m=2000,
        thermal_distance_std_m=1000,
        thermal_distance_max_m=10000,
    )

    plot_flight_conditions(flight_conditions, path_file=path_file, show=show)


if __name__ == "__main__":
    plot_conditions()
