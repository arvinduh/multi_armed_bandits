"""An implementation of the Epsilon-Greedy multi-arm bandit algorithm."""

import numpy as np

from src.algorithms import base


class EpsilonGreedy(base.MultiArmAlgorithm):
  """An Epsilon-Greedy algorithm.

  This strategy balances exploration and exploitation. With a probability of
  epsilon, it chooses a random arm (explores). With a probability of
  1-epsilon, it chooses the arm with the highest observed average reward
  (exploits).
  """

  def __init__(self, num_options: int, epsilon: float):
    """Initializes the Epsilon-Greedy algorithm.

    Args:
      num_options: The number of arms (options) available.
      epsilon: The probability of choosing a random arm (exploring), typically a
        small value between 0.0 and 1.0.
    """
    super().__init__(num_options)
    if not 0.0 <= epsilon <= 1.0:
      raise ValueError("epsilon must be between 0.0 and 1.0.")
    self.epsilon = epsilon

  def select_arm(self) -> int:
    """Selects an arm using the epsilon-greedy strategy.

    Returns:
      The index of the arm to pull.
    """
    # Generate a random number to decide between exploration and exploitation.
    if np.random.rand() < self.epsilon:
      # --- Explore ---
      # Choose any arm with equal probability.
      return np.random.randint(self.num_options)
    else:
      # --- Exploit ---
      return int(np.argmax(self.average_rewards))
