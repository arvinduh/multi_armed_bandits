"""Abstract base classes for multi-arm bandit algorithms."""

import abc

import numpy as np


class MultiArmAlgorithm(abc.ABC):
  """Abstract base class for multi-arm bandit algorithms."""

  def __init__(self, num_options: int) -> None:
    """Initializes the multi-arm algorithm.

    Args:
      num_options: The number of arms (options) available to the algorithm.
    """
    if not isinstance(num_options, int) or num_options <= 0:
      raise ValueError("num_options must be a positive integer.")

    self.num_options = num_options
    self.period = 0
    self.arm_counts = np.zeros(num_options, dtype=np.int32)
    self.arm_rewards = np.zeros(num_options, dtype=np.float64)

  @property
  def average_rewards(self) -> np.ndarray:
    """Calculates the average reward for each arm, handling division by zero."""
    # Create a mask to avoid division by zero where counts are zero.
    non_zero_counts = self.arm_counts > 0
    averages = np.zeros(self.num_options, dtype=np.float64)
    averages[non_zero_counts] = (
      self.arm_rewards[non_zero_counts] / self.arm_counts[non_zero_counts]
    )
    return averages

  @abc.abstractmethod
  def select_arm(self) -> int:
    """Selects an arm to pull based on the algorithm's strategy."""
    raise NotImplementedError

  def update(self, chosen_arm: int, reward: float) -> None:
    """Updates the algorithm's state with the result of a single pull.

    Args:
      chosen_arm: The index of the arm that was pulled.
      reward: The reward received from pulling the chosen arm.
    """
    if not 0 <= chosen_arm < self.num_options:
      raise IndexError(
        f"chosen_arm {chosen_arm} is out of range [0, {self.num_options - 1}]."
      )

    self.arm_counts[chosen_arm] += 1
    self.arm_rewards[chosen_arm] += reward
    self.period += 1

  def reset(self) -> None:
    """Resets the internal state for a new simulation."""
    self.period = 0
    self.arm_counts.fill(0)
    self.arm_rewards.fill(0.0)

  def get_total_reward(self) -> float:
    """Calculates the total reward accumulated across all arms."""
    return np.sum(self.arm_rewards)
