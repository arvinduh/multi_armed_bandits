"""
An implementation of a custom bandit algorithm that explores each arm until its
running standard deviation stabilizes, then exploits the best-found arm.
"""

import numpy as np

from src.algorithms import base


class StableExploration(base.MultiArmAlgorithm):
  """
  An algorithm that explores each arm sequentially until its running standard
  deviation has stabilized for a set number of consecutive pulls. Once all arms
  are stabilized, it commits to exploiting the arm with the highest mean.
  """

  def __init__(
    self,
    num_options: int,
    stability_threshold: float = 0.5,
    consecutive_count_needed: int = 3,
  ):
    """Initializes the StableExploration algorithm."""
    super().__init__(num_options)

    # This algorithm exclusively manages its own reward history.
    self.reward_history: list[list[float]] = [[] for _ in range(num_options)]

    if stability_threshold < 0:
      raise ValueError("stability_threshold must be non-negative.")
    if consecutive_count_needed < 1:
      raise ValueError("consecutive_count_needed must be at least 1.")

    self.stability_threshold = stability_threshold
    self.consecutive_count_needed = consecutive_count_needed
    # Call reset to initialize the algorithm's specific state
    self.reset()

  def select_arm(self) -> int:
    """Selects an arm based on the current phase (exploration or exploitation)."""
    # Phase 2: Exploitation
    if self._optimal_choice is not None:
      return self._optimal_choice

    # Phase 1: Exploration
    return self._current_arm_to_explore

  def update(self, chosen_arm: int, reward: float) -> None:
    """Updates state and checks for stabilization, managing its own history."""
    # First, append the reward to its own history list.
    self.reward_history[chosen_arm].append(reward)

    # Then, call the parent method to update the standard aggregates.
    super().update(chosen_arm, reward)

    # Only perform stabilization checks during the exploration phase.
    if self._optimal_choice is not None:
      return

    history = self.reward_history[chosen_arm]
    if len(history) < 2:
      return

    new_std = np.std(history, ddof=1)
    old_std = self._last_running_std[chosen_arm]

    if abs(new_std - old_std) < self.stability_threshold:
      self._consecutive_stable_pulls += 1
    else:
      self._consecutive_stable_pulls = 0  # Reset counter if not stable.

    self._last_running_std[chosen_arm] = new_std

    if self._consecutive_stable_pulls >= self.consecutive_count_needed:
      self._move_to_next_arm()

  def _move_to_next_arm(self):
    """Moves the exploration focus to the next arm or enters exploitation phase."""
    self._current_arm_to_explore += 1
    self._consecutive_stable_pulls = 0

    if self._current_arm_to_explore >= self.num_options:
      # Enter Phase 2: Exploitation
      self._optimal_choice = int(np.argmax(self.average_rewards))

  def reset(self) -> None:
    """Resets the algorithm to its initial state for a new simulation."""
    super().reset()
    self.reward_history = [[] for _ in range(self.num_options)]
    self._current_arm_to_explore: int = 0
    self._consecutive_stable_pulls: int = 0
    self._last_running_std: np.ndarray = np.zeros(self.num_options)
    self._optimal_choice: int | None = None
