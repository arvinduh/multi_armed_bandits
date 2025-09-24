"""
An implementation of a multi-armed bandit algorithm that uses a pre-trained
neural network to select arms.
"""

import os

import numpy as np

from src import train
from src.algorithms import base

# The model is now hard-locked to solve a 3-armed bandit problem.
MAX_ARMS = 3


class NeuralBandit(base.MultiArmAlgorithm):
  """
  A bandit algorithm that uses a trained neural network to make decisions.
  """

  def __init__(self, num_options: int, total_periods: int, weights_path: str):
    """Initializes the NeuralBandit algorithm."""
    super().__init__(num_options)
    if num_options != MAX_ARMS:
      raise ValueError(f"This model is hard-locked to exactly {MAX_ARMS} arms.")
    if not os.path.exists(weights_path):
      raise FileNotFoundError(f"Weights file not found at {weights_path}")

    self.total_periods = total_periods
    # State for efficient standard deviation calculation.
    self.arm_rewards_sum_sq = np.zeros(num_options)

    # Build the model architecture and load the pre-trained weights.
    self.model = train.build_model(MAX_ARMS)
    self.model.load_weights(weights_path)

  def select_arm(self) -> int:
    """Selects an arm by getting a prediction from the neural network."""
    # 1. Construct the feature vector representing the current state.
    running_stds = np.zeros(self.num_options)
    # Only calculate std dev for arms pulled more than once.
    valid_std_mask = self.arm_counts > 1
    if np.any(valid_std_mask):
      valid_counts = self.arm_counts[valid_std_mask]
      # We use self.average_rewards which is pre-calculated in the base class.
      valid_avg_rewards = self.average_rewards[valid_std_mask]
      valid_sums_sq = self.arm_rewards_sum_sq[valid_std_mask]

      variances = (valid_sums_sq / valid_counts) - (valid_avg_rewards**2)
      variances[variances < 0] = 0  # Correct for floating point errors.
      running_stds[valid_std_mask] = np.sqrt(variances)

    # The model expects a flat vector with 11 features for the 3-arm case.
    feature_vector = np.concatenate(
      [
        [self.period, self.total_periods],
        self.arm_counts,
        self.average_rewards,
        running_stds,
      ]
    ).astype(np.float32)

    # 2. Get a prediction from the model.
    feature_batch = np.expand_dims(feature_vector, axis=0)
    probabilities = self.model.predict(feature_batch, verbose=0)[0]

    # 3. Choose the arm with the highest probability.
    return int(np.argmax(probabilities))

  def update(self, chosen_arm: int, reward: float) -> None:
    """Updates the aggregates and sum of squares for std dev calculation."""
    # Update base class aggregates first (counts, averages, period).
    super().update(chosen_arm, reward)
    # Update the sum of squares for our internal calculation.
    self.arm_rewards_sum_sq[chosen_arm] += reward**2

  def reset(self) -> None:
    """Resets the state for a new simulation."""
    super().reset()
    self.arm_rewards_sum_sq.fill(0)
