"""An implementation of an explore-then-exploit multi-arm bandit algorithm."""

import numpy as np

from src.algorithms import base


class ExploreExploit(base.MultiArmAlgorithm):
  """An explore-then-exploit algorithm.

  This strategy first explores each arm once to gather initial data, then
  exclusively exploits the arm with the highest observed average reward for all
  subsequent periods.
  """

  def __init__(self, num_options: int) -> None:
    """Initializes the explore-then-exploit algorithm."""
    super().__init__(num_options)
    self._optimal_choice: int | None = None

  def select_arm(self) -> int:
    """Selects an arm, exploring each sequentially then exploiting the best.

    Returns:
      The index of the arm to pull.
    """
    # Exploration phase: pull each arm once.
    if self.period < self.num_options:
      return self.period

    # Exploitation phase: consistently pull the best known arm.
    if self._optimal_choice is None:
      self._optimal_choice = int(np.argmax(self.arm_rewards))

    return self._optimal_choice

  def reset(self) -> None:
    """Resets the algorithm and clears the identified optimal choice."""
    super().reset()
    self._optimal_choice = None
