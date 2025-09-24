"""
Plots the changes in standard deviation for each arm during stable exploration.
This module creates visualizations showing how the running standard deviation
changes from the previous period for each arm when using the stable exploration algorithm.
"""

import os
import sys
from typing import Any, Callable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.algorithms import stable


def track_std_evolution(
  algorithm: stable.StableExploration,
  distributions: Sequence[Tuple[Callable[..., float], Tuple[Any, ...]]],
  periods: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Runs a single simulation and tracks the standard deviation changes for each arm.

  Args:
      algorithm: The stable exploration algorithm instance
      distributions: Distribution functions and their parameters for each arm
      periods: Number of time periods to simulate

  Returns:
      Tuple of (time_points, std_changes) where:
      - time_points: Array of time steps (starting from period 2)
      - std_changes: Array of shape (periods-1, num_arms) with std dev changes from previous period
  """
  algorithm.reset()
  num_arms = len(distributions)

  # Track standard deviation for each arm at each time point
  std_evolution = np.zeros((periods, num_arms))

  for period in range(periods):
    # Select arm and get reward
    arm = algorithm.select_arm()

    # Get reward sample from the chosen distribution
    func, args = distributions[arm]
    reward = func(*args)

    # Update algorithm
    algorithm.update(chosen_arm=arm, reward=reward)

    # Record current running standard deviation for each arm
    for arm_idx in range(num_arms):
      if len(algorithm.reward_history[arm_idx]) >= 2:
        std_evolution[period, arm_idx] = np.std(
          algorithm.reward_history[arm_idx], ddof=1
        )
      else:
        # If less than 2 samples, std is 0 or use previous value
        std_evolution[period, arm_idx] = (
          std_evolution[period - 1, arm_idx] if period > 0 else 0
        )

  # Calculate differences from the previous period (skip first period)
  std_changes = np.zeros((periods - 1, num_arms))
  for period in range(1, periods):
    std_changes[period - 1, :] = (
      std_evolution[period, :] - std_evolution[period - 1, :]
    )

  time_points = np.arange(
    2, periods + 1
  )  # Start from period 2 since we need differences
  return time_points, std_changes


def plot_stable_std_evolution(
  distributions_params: Sequence[Tuple[float, float]],
  periods: int = 300,
  stability_threshold: float = 0.5,
  consecutive_count: int = 3,
  save_path: str = None,
) -> None:
  """
  Creates and displays a plot showing the changes in standard deviation
  from the previous period for each arm during stable exploration.

  Args:
      distributions_params: List of (mean, std_dev) tuples for each arm
      periods: Number of time periods to simulate
      stability_threshold: Stability threshold for the algorithm
      consecutive_count: Number of consecutive stable pulls needed
      save_path: Optional path to save the plot
  """
  # Convert distribution parameters to distribution functions
  distributions = [
    (norm.rvs, (mean, std)) for mean, std in distributions_params
  ]

  # Initialize the stable exploration algorithm
  num_arms = len(distributions)
  algorithm = stable.StableExploration(
    num_options=num_arms,
    stability_threshold=stability_threshold,
    consecutive_count_needed=consecutive_count,
  )

  # Track standard deviation changes
  time_points, std_changes = track_std_evolution(
    algorithm, distributions, periods
  )

  # Create the plot
  plt.figure(figsize=(12, 8))

  # Plot standard deviation changes for each arm
  colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green
  arm_names = [
    f"Arm {i + 1}: N({mean}, {std})"
    for i, (mean, std) in enumerate(distributions_params)
  ]

  for arm_idx in range(num_arms):
    plt.plot(
      time_points,
      std_changes[:, arm_idx],
      label=arm_names[arm_idx],
      color=colors[arm_idx],
      linewidth=2,
      alpha=0.8,
    )

  # Add horizontal line at y=0 for reference
  plt.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

  # Customize the plot
  plt.xlabel("Time Period", fontsize=12)
  plt.ylabel("Change in Standard Deviation from Previous Period", fontsize=12)
  plt.title(
    f"Standard Deviation Changes During Stable Exploration\n"
    f"(Threshold: {stability_threshold}, Consecutive: {consecutive_count})",
    fontsize=14,
    fontweight="bold",
  )
  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()

  # Add text box with algorithm info
  info_text = (
    f"Stability Threshold: {stability_threshold}\n"
    f"Consecutive Count: {consecutive_count}\n"
    f"Total Periods: {periods}\n"
    f"Note: Shows change from previous period"
  )
  plt.text(
    0.02,
    0.98,
    info_text,
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
  )

  # Save the plot if path is provided
  if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

  plt.show()


def plot_multiple_runs_std_evolution(
  distributions_params: Sequence[Tuple[float, float]],
  num_runs: int = 5,
  periods: int = 300,
  stability_threshold: float = 0.5,
  consecutive_count: int = 3,
  save_path: str = None,
) -> None:
  """
  Creates a plot showing the standard deviation changes across multiple runs
  to show the variability in the algorithm's behavior.

  Args:
      distributions_params: List of (mean, std_dev) tuples for each arm
      num_runs: Number of simulation runs to overlay
      periods: Number of time periods to simulate
      stability_threshold: Stability threshold for the algorithm
      consecutive_count: Number of consecutive stable pulls needed
      save_path: Optional path to save the plot
  """
  # Convert distribution parameters to distribution functions
  distributions = [
    (norm.rvs, (mean, std)) for mean, std in distributions_params
  ]
  num_arms = len(distributions)

  plt.figure(figsize=(15, 10))

  # Colors for each arm
  colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
  arm_names = [
    f"Arm {i + 1}: N({mean}, {std})"
    for i, (mean, std) in enumerate(distributions_params)
  ]

  # Run multiple simulations
  for run in range(num_runs):
    # Initialize the algorithm for this run
    algorithm = stable.StableExploration(
      num_options=num_arms,
      stability_threshold=stability_threshold,
      consecutive_count_needed=consecutive_count,
    )

    # Track changes for this run
    time_points, std_changes = track_std_evolution(
      algorithm, distributions, periods
    )

    # Plot each arm with transparency
    for arm_idx in range(num_arms):
      alpha = 0.7 if run == 0 else 0.3  # First run more opaque
      label = arm_names[arm_idx] if run == 0 else None  # Only label first run

      plt.plot(
        time_points,
        std_changes[:, arm_idx],
        color=colors[arm_idx],
        alpha=alpha,
        linewidth=1.5 if run == 0 else 1,
        label=label,
      )

  # Add horizontal line at y=0 for reference
  plt.axhline(
    y=0,
    color="black",
    linestyle="-",
    alpha=0.5,
    linewidth=1.5,
    label="No Change",
  )

  # Customize the plot
  plt.xlabel("Time Period", fontsize=12)
  plt.ylabel("Change in Standard Deviation from Previous Period", fontsize=12)
  plt.title(
    f"Standard Deviation Changes Across {num_runs} Runs - Stable Exploration\n"
    f"(Threshold: {stability_threshold}, Consecutive: {consecutive_count})",
    fontsize=14,
    fontweight="bold",
  )
  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.grid(True, alpha=0.3)
  plt.tight_layout()

  # Save the plot if path is provided
  if save_path:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Multi-run plot saved to: {save_path}")

  plt.show()


if __name__ == "__main__":
  # Default distribution parameters from the simulation results
  distributions_params = [(12, 4), (10, 6), (8, 4)]

  # Create single run plot
  print("Creating single run standard deviation evolution plot...")
  plot_stable_std_evolution(
    distributions_params=distributions_params,
    periods=300,
    stability_threshold=0.5,
    consecutive_count=3,
    save_path="img/stable_std_evolution_single.png",
  )

  # Create multiple runs plot
  print("Creating multiple runs standard deviation evolution plot...")
  plot_multiple_runs_std_evolution(
    distributions_params=distributions_params,
    num_runs=5,
    periods=300,
    stability_threshold=0.5,
    consecutive_count=3,
    save_path="img/stable_std_evolution_multiple.png",
  )
