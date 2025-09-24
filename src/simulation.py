"""A Monte Carlo simulation runner for multi-arm bandit algorithms."""

import inspect
from typing import Any, Callable, Sequence

import numpy as np
from tqdm import tqdm

from src.algorithms import base


class MonteCarlo:
  """Runs Monte Carlo simulations for a given multi-arm bandit algorithm."""

  def __init__(
    self,
    model: base.MultiArmAlgorithm,
    distributions: Sequence[tuple[Callable[..., float], tuple[Any, ...]]],
    periods: int,
    num_simulations: int,
  ):
    """Initializes the Monte Carlo simulation.

    Args:
      model: The multi-arm algorithm to run.
      distributions: A sequence of tuples, where each tuple contains a callable
        distribution function and a tuple of its arguments.
      periods: The number of periods (time steps) in a single simulation.
      num_simulations: The number of simulations to run.
    """
    if not isinstance(periods, int) or periods <= 0:
      raise ValueError("periods must be a positive integer.")
    if not isinstance(num_simulations, int) or num_simulations <= 0:
      raise ValueError("num_simulations must be a positive integer.")

    self._validate_distributions(distributions)

    self.model = model
    self.distributions = distributions
    self.periods = periods
    self.num_simulations = num_simulations

  def _validate_distributions(
    self, distributions: Sequence[tuple[Callable[..., float], tuple[Any, ...]]]
  ) -> None:
    """Checks if the provided distributions are valid."""
    if not isinstance(distributions, Sequence):
      raise TypeError("distributions must be a sequence.")
    if not distributions:
      raise ValueError("distributions cannot be empty.")

    for i, item in enumerate(distributions):
      if not isinstance(item, tuple) or len(item) != 2:
        raise TypeError(
          f"distributions[{i}] must be a tuple of (function, args)."
        )

      func, args = item
      if not callable(func):
        raise TypeError(f"distributions[{i}][0] must be a callable function.")
      if not isinstance(args, tuple):
        raise TypeError(f"distributions[{i}][1] must be a tuple of arguments.")

      try:
        inspect.signature(func).bind(*args)
      except TypeError as e:
        raise ValueError(
          f"distributions[{i}]: function {func.__name__} cannot accept"
          f" arguments {args}: {e}"
        )

  def _get_reward_sample(self, choice: int) -> float:
    """Gets a single reward sample from the chosen distribution.

    Args:
      choice: The index of the distribution to sample from.

    Returns:
      A float reward value from the distribution.
    """

    func, args = self.distributions[choice]
    return func(*args)

  def run(self) -> tuple[float, float]:
    """Runs all Monte Carlo simulations and computes the average reward.

    Returns:
        A tuple containing the mean and standard deviation of the total reward
        across all simulations.
    """
    sum_of_rewards = 0.0
    sum_of_squared_rewards = 0.0
    n = self.num_simulations

    if n == 0:
      return 0.0, 0.0

    for _ in tqdm(range(n), desc="Running Simulations"):
      self.model.reset()
      for _ in range(self.periods):
        arm = self.model.select_arm()
        reward = self._get_reward_sample(arm)
        self.model.update(chosen_arm=arm, reward=reward)

      # Get the total reward for this completed simulation.
      total_reward = self.model.get_total_reward()

      # Update the two accumulators for the single-pass calculation.
      sum_of_rewards += total_reward
      sum_of_squared_rewards += total_reward**2

    # Calculate mean and variance using the efficient single-pass formula.
    mean = sum_of_rewards / n
    # Var(X) = E[X^2] - (E[X])^2
    variance = (sum_of_squared_rewards / n) - mean**2

    # Handle potential floating-point inaccuracies where variance is slightly < 0.
    std_dev = np.sqrt(max(0, variance))

    return mean, std_dev
