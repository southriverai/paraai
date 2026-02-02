from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from paraai.experiment import ExperimentOutput, ExperimentOutputBatch
from paraai.flight_conditions import FlightConditions


def get_statistics(experiment_output_batch: ExperimentOutputBatch):
    distances_km = []
    for experiment_output in experiment_output_batch.list_experiment_outputs:
        distances_km.append(experiment_output.flight_state.list_distance_m[-1] / 1000.0)
    p50_distance = np.percentile(distances_km, 50)  # round to 2 decimal places
    p50_distance = round(p50_distance, 2)
    p90_distance = np.percentile(distances_km, 90)
    p90_distance = round(p90_distance, 2)
    return {
        "p50_distance": p50_distance,
        "p90_distance": p90_distance,
    }


def print_statistics(experiment_result_baches: list[ExperimentOutputBatch], labels: list[str]):
    for experiment_output_batch, label in zip(experiment_result_baches, labels):
        stats = get_statistics(experiment_output_batch)
        print(label)
        print("p50 distance", stats["p50_distance"])
        print("p90 distance", stats["p90_distance"])


def plot_flight_hists(
    experiment_result_baches: list[ExperimentOutputBatch],
    labels: list[str],
    path_file: Optional[Path] = None,
    show: bool = False,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    best_p50_policy_name = None
    best_p50_distance = 0
    best_p90_policy_name = None
    best_p90_distance = 0
    for experiment_output_batch, label in zip(experiment_result_baches, labels):
        print(label)
        distances_km = []
        durations_s = []
        for experiment_output in experiment_output_batch.list_experiment_outputs:
            distances_km.append(experiment_output.flight_state.list_distance_m[-1] / 1000.0)
            durations_s.append(experiment_output.flight_state.list_time_s[-1])

        p50_distance = np.percentile(distances_km, 50)  # round to 2 decimal places
        p50_distance = round(p50_distance, 2)
        p90_distance = np.percentile(distances_km, 90)
        p90_distance = round(p90_distance, 2)
        print("p50 distance", p50_distance)
        print("p90 distance", p90_distance)
        if p50_distance > best_p50_distance:
            best_p50_distance = p50_distance
            best_p50_policy_name = label
        if p90_distance > best_p90_distance:
            best_p90_distance = p90_distance
            best_p90_policy_name = label
        axes[0].hist(
            distances_km,
            bins=20,
            label=label,
            alpha=0.7,
        )
        axes[1].hist(
            durations_s,
            bins=20,
            label=label,
            alpha=0.7,
        )
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distance Distribution")

    axes[1].set_xlabel("Duration (s)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Duration Distribution")
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()

    # best p50 p
    print("best p50 policy", best_p50_policy_name)
    print("best p90 policy", best_p90_policy_name)
    if show:
        plt.show()
    if path_file is not None:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_file)
        print(f"Flight hists plot saved to {path_file}")


def plot_flight_hists_2(
    experiment_result_baches: list[ExperimentOutputBatch],
    labels: list[str],
    path_file: Optional[Path] = None,
    show: bool = False,
):
    fig, ax_distance = plt.subplots(1, 1, figsize=(12, 6))

    # First pass: collect all data to determine bin ranges
    all_distances_km = []
    for experiment_output_batch in experiment_result_baches:
        for experiment_output in experiment_output_batch.list_experiment_outputs:
            all_distances_km.append(experiment_output.flight_state.list_distance_m[-1] / 1000.0)

    # Create bins based on all data
    num_bins = 100
    _, distance_bins = np.histogram(all_distances_km, bins=num_bins)  # Get bin edges

    # Second pass: plot histograms using the same bins
    flight_count = 0
    for experiment_output_batch, label in zip(experiment_result_baches, labels):
        distances_km = []
        for experiment_output in experiment_output_batch.list_experiment_outputs:
            distances_km.append(experiment_output.flight_state.list_distance_m[-1] / 1000.0)
        flight_count = len(distances_km)
        ax_distance.hist(
            distances_km,
            bins=distance_bins,
            label=label,
            alpha=0.7,
        )
    ax_distance.set_xlabel("Distance (km)")
    ax_distance.set_ylabel("Frequency")
    ax_distance.set_title(f"Distance histogram over {flight_count} flights")
    ax_distance.legend()

    plt.tight_layout()
    if show:
        plt.show()
    if path_file is not None:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_file)
        print(f"Flight hists plot saved to {path_file}")


def plot_flight_hist(experiment_output: ExperimentOutput):
    plt.figure(figsize=(12, 6))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].hist(experiment_output.flight_state.list_distance_m, bins=20)
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distance Distribution")
    axes[1].hist(experiment_output.flight_state.list_time_s, bins=20)
    axes[1].set_xlabel("Duration (s)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Duration Distribution")
    plt.tight_layout()
    plt.show()


def plot_flight(
    experiment_result: ExperimentOutput,
    path_file: Optional[Path] = None,
    show: bool = False,
):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    flight_state = experiment_result.flight_state
    termals = experiment_result.termals
    # Convert distances to kilometers
    distance_km = [d / 1000.0 for d in flight_state.list_distance_m]
    distance_max_km = flight_state.list_distance_m[-1] / 1000.0

    # plot distance vs time
    axes[0].plot(distance_km, flight_state.list_time_s)
    axes[0].set_xlabel("Distance (km)")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Distance vs Time")
    axes[0].set_xlim(left=0, right=distance_max_km)
    axes[0].grid(True)

    # plot distance vs altitude
    axes[1].plot(distance_km, flight_state.list_altitude_m)
    axes[1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Distance vs Altitude")
    axes[1].set_xlim(left=0, right=distance_max_km)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True)

    # plot flight conditions (thermal strength vs distance)
    distance_series, thermal_strength_series = FlightConditions.series_termal(termals, flight_state.list_distance_m[-1])
    # Convert distance from meters to kilometers for better readability
    distance_series_km = [d / 1000.0 for d in distance_series]

    axes[2].step(
        distance_series_km,
        thermal_strength_series,
        where="post",
        linewidth=2,
        color="blue",
        label="Thermal Strength",
        alpha=0.8,
    )
    axes[2].fill_between(
        distance_series_km,
        thermal_strength_series,
        step="post",
        alpha=0.3,
        color="blue",
        label="_nolegend_",
    )
    axes[2].set_xlabel("Distance (km)")
    axes[2].set_ylabel("Thermal Strength (m/s)")
    axes[2].set_title(f"Thermal Conditions Along Flight Path (0 to {distance_max_km:.1f} km)")
    axes[2].set_xlim(left=0, right=distance_max_km)
    axes[2].set_ylim(bottom=0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    if show:
        plt.show()
    if path_file is not None:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_file)
        print(f"Flight plot saved to {path_file}")


def plot_flight_conditions(
    flight_conditions: FlightConditions,
    path_file: Optional[Path] = None,
    show: bool = False,
):
    """
    Plot thermal strengths along the flight distance from 0 to max distance.
    Uses the series_termal function to get blocky series for plotting.
    """
    # Get blocky series from series_termal function
    distance_series, thermal_strength_series = flight_conditions.series_termal_sampled()

    # Convert distance from meters to kilometers
    distance_series_km = [d / 1000.0 for d in distance_series]
    distance_max_km = flight_conditions.distance_max_m / 1000.0

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the step-like series (blocky/step plot)
    # Using step='post' to make the steps appear at the right positions
    ax.step(
        distance_series_km,
        thermal_strength_series,
        where="post",
        linewidth=2,
        color="blue",
        label="Thermal Strength",
        alpha=0.8,
    )

    # Fill the area under the curve for better visualization
    ax.fill_between(
        distance_series_km,
        thermal_strength_series,
        step="post",
        alpha=0.3,
        color="blue",
        label="_nolegend_",
    )

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Thermal Strength (m/s)")
    ax.set_title(f"Thermal Conditions Along Flight Path (0 to {distance_max_km:.1f} km)")
    ax.set_xlim(left=0, right=distance_max_km)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if path_file is not None:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_file)
        print(f"Flight conditions plot saved to {path_file}")
    if show:
        plt.show()
