"""Trains a neural network to solve the multi-armed bandit problem.

This script uses an online, batch-based learning approach with Experience Replay,
a core technique from Deep Reinforcement Learning, to train a model that learns
the optimal action (i.e., which arm to pull) based on evolving statistics.
"""

import collections
import os
import random
from collections.abc import Sequence

import numpy as np
import tensorflow as tf
from absl import app, flags
from tqdm import tqdm

# The model is hard-locked to solve a 3-armed bandit problem.
_MAX_ARMS = 3

FLAGS = flags.FLAGS
flags.DEFINE_integer(
  name="num_episodes",
  default=1000,
  help="Number of bandit problem episodes to train on.",
)
flags.DEFINE_integer(
  name="batch_size", default=128, help="Training batch size."
)
flags.DEFINE_float(
  name="epsilon_start",
  default=1.0,
  help="Starting value for epsilon-greedy exploration.",
)
flags.DEFINE_float(
  name="epsilon_end",
  default=0.01,
  help="Minimum value for epsilon-greedy exploration.",
)
flags.DEFINE_float(
  name="epsilon_decay_rate",
  default=0.995,
  help="Decay rate for epsilon. Higher is slower.",
)
flags.DEFINE_integer(
  name="info_freq", default=20, help="Display info every N episodes."
)
flags.DEFINE_integer(
  name="checkpoint_freq",
  default=100,
  help="Save a model checkpoint every N episodes.",
)
flags.DEFINE_integer(
  name="eval_episodes",
  default=100,
  help="Number of episodes to run for each evaluation.",
)
flags.DEFINE_string(
  name="resume_from",
  default=None,
  help="Path to weights file to resume training from.",
)


class ReplayBuffer:
  """A simple FIFO experience replay buffer."""

  def __init__(self, capacity: int):
    """Initializes the ReplayBuffer.

    Args:
      capacity: The maximum number of experiences to store in the buffer.
    """
    self._buffer = collections.deque(maxlen=capacity)

  def add(self, experience: tuple[np.ndarray, int]) -> None:
    """Adds an experience to the buffer."""
    self._buffer.append(experience)

  def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Samples a random batch of experiences from the buffer.

    Args:
      batch_size: The number of experiences to sample.

    Returns:
      A tuple of (features, labels) if the buffer has enough samples,
      otherwise None.
    """
    if len(self._buffer) < batch_size:
      return None
    samples = random.sample(self._buffer, batch_size)
    features, labels = zip(*samples)
    return np.array(features), np.array(labels)


def build_model(num_arms: int, learning_rate: float = 0.001) -> tf.keras.Model:
  """Builds and compiles a Keras MLP model for the bandit problem."""
  input_size = 2 + (num_arms * 3)  # period, total_periods, counts, avgs, stds

  inputs = tf.keras.layers.Input(shape=(input_size,))
  x = tf.keras.layers.LayerNormalization()(inputs)
  x = tf.keras.layers.Dense(128, activation="relu")(x)
  x = tf.keras.layers.Dense(128, activation="relu")(x)
  outputs = tf.keras.layers.Dense(num_arms, activation="softmax")(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.optimizer = tf.keras.optimizers.Adam(learning_rate)
  model.loss = tf.keras.losses.SparseCategoricalCrossentropy()
  return model


@tf.function
def train_step(
  model: tf.keras.Model, features: tf.Tensor, labels: tf.Tensor
) -> tf.Tensor:
  """Performs a single batch training step."""
  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = model.loss(y_true=labels, y_pred=predictions)

  grads = tape.gradient(loss, model.trainable_variables)
  model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss


def evaluate_model(model: tf.keras.Model) -> float:
  """Evaluates the model's performance with exploration turned off."""
  total_optimal_choices = 0
  total_steps = 0
  num_arms = _MAX_ARMS

  for _ in tqdm(range(FLAGS.eval_episodes), desc="  Evaluating", leave=False):
    total_periods = np.random.randint(100, 501)
    means = np.random.uniform(-10, 10, size=num_arms)
    stds = np.random.uniform(1, 5, size=num_arms)
    optimal_action = np.argmax(means)

    arm_counts = np.zeros(num_arms)
    arm_rewards_sum = np.zeros(num_arms)

    for period in range(total_periods):
      current_avg_rewards = np.zeros(num_arms)
      non_zero_counts = arm_counts > 0
      current_avg_rewards[non_zero_counts] = (
        arm_rewards_sum[non_zero_counts] / arm_counts[non_zero_counts]
      )

      feature_vector = np.concatenate(
        [
          [period, total_periods],
          arm_counts,
          current_avg_rewards,
          np.zeros(num_arms),  # Std dev not needed for greedy evaluation
        ]
      ).astype(np.float32)

      predictions = model(np.array([feature_vector]), training=False)
      action = np.argmax(predictions[0])

      if action == optimal_action:
        total_optimal_choices += 1
      total_steps += 1

      reward = np.random.normal(means[action], stds[action])
      arm_counts[action] += 1
      arm_rewards_sum[action] += reward

  return (total_optimal_choices / total_steps) * 100 if total_steps > 0 else 0.0


def train_online(model: tf.keras.Model, model_dir: str) -> None:
  """Trains the model using an experience replay buffer."""
  print(f"Starting training for {FLAGS.num_episodes} episodes...")
  replay_buffer = ReplayBuffer(capacity=50_000)
  epsilon = FLAGS.epsilon_start
  total_loss = 0.0
  num_training_steps = 0
  num_arms = _MAX_ARMS

  for episode_num in tqdm(range(FLAGS.num_episodes), desc="Episodes"):
    total_periods = np.random.randint(100, 501)
    means = np.random.uniform(-10, 10, size=num_arms)
    stds = np.random.uniform(1, 5, size=num_arms)
    optimal_action = np.argmax(means)

    # Statistics for Welford's online algorithm
    arm_counts = np.zeros(num_arms, dtype=np.int32)
    arm_means = np.zeros(num_arms, dtype=np.float32)
    arm_m2s = np.zeros(num_arms, dtype=np.float32)  # Sum of squares of diffs

    for period in range(total_periods):
      running_stds = np.zeros(num_arms)
      valid_std_mask = arm_counts > 1
      if np.any(valid_std_mask):
        variances = arm_m2s[valid_std_mask] / (arm_counts[valid_std_mask] - 1)
        variances[variances < 0] = 0  # Correct for floating point errors
        running_stds[valid_std_mask] = np.sqrt(variances)

      feature_vector = np.concatenate(
        [
          [period, total_periods],
          arm_counts,
          arm_means,  # Use the running mean directly
          running_stds,
        ]
      ).astype(np.float32)

      if np.random.rand() < epsilon:
        action = np.random.randint(num_arms)
      else:
        predictions = model(np.array([feature_vector]), training=False)
        action = np.argmax(predictions[0])

      replay_buffer.add((feature_vector, optimal_action))

      batch = replay_buffer.sample(FLAGS.batch_size)
      if batch:
        features, labels = batch
        loss = train_step(
          model,
          features=tf.constant(features),
          labels=tf.constant(labels, dtype=tf.int32),
        )
        total_loss += loss
        num_training_steps += 1

      # Get reward and update statistics using Welford's algorithm
      reward = np.random.normal(means[action], stds[action])
      arm_counts[action] += 1
      count = arm_counts[action]
      delta = reward - arm_means[action]
      arm_means[action] += delta / count
      delta2 = reward - arm_means[action]
      arm_m2s[action] += delta * delta2

    epsilon = max(FLAGS.epsilon_end, epsilon * FLAGS.epsilon_decay_rate)

    if (episode_num + 1) % FLAGS.info_freq == 0:
      eval_accuracy = evaluate_model(model)
      running_avg_loss = (
        total_loss / num_training_steps if num_training_steps > 0 else 0.0
      )
      tqdm.write(
        f"  Episode {episode_num + 1:04d} | "
        f"Eval Accuracy: {eval_accuracy:6.2f}% | "
        f"Running Avg Loss: {running_avg_loss:.4f} | "
        f"Epsilon: {epsilon:.3f}"
      )
    if (episode_num + 1) % FLAGS.checkpoint_freq == 0:
      path = os.path.join(
        model_dir, f"bandit_nn_episode_{(episode_num + 1):04d}.weights.h5"
      )
      model.save_weights(path)
      tqdm.write(f"  >> Checkpoint saved to {path}")


def main(argv: Sequence[str]) -> None:
  """Main training script function."""
  del argv  # Unused.
  model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
  os.makedirs(model_dir, exist_ok=True)

  model = build_model(_MAX_ARMS)
  if FLAGS.resume_from:
    print(f"Resuming training from {FLAGS.resume_from}")
    model.load_weights(FLAGS.resume_from)

  train_online(model, model_dir)

  final_path = os.path.join(model_dir, "bandit_nn_final.weights.h5")
  model.save_weights(final_path)
  print(f"\nFinal model weights saved to {final_path}")


if __name__ == "__main__":
  app.run(main)
