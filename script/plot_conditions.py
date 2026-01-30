from paraai.flight_conditions import FlightConditions
from paraai.tools_plot import plot_flight_conditions

flight_conditions = FlightConditions(
    take_off_time_s=0,
    take_off_altitude_m=1000,
    landing_time_s=3600 * 6,
    distance_max_m=3600 * 10,
    termal_ceiling_m=1000,
    termal_net_climb_min_m_s=0.1,
    termal_net_climb_mean_m_s=0.5,
    termal_net_climb_std_m_s=0.5,
    termal_net_climb_max_m_s=8,
    termal_distance_min_m=1000,
    termal_distance_mean_m=2000,
    termal_distance_std_m=1000,
    termal_distance_max_m=10000,
)

plot_flight_conditions(flight_conditions)
