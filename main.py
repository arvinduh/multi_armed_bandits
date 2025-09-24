# Arvin Duh
# 2025-09-24

"""Main application entry point for running multi-arm bandit simulations."""

import datetime
import json

from absl import app, flags
from scipy.stats import norm

from src import simulation
from src.algorithms import epsilon, explore, neural, stable

# --- Define Constants and Flags ---

FLAGS = flags.FLAGS
RESULTS_FILE = "simulation_results.jsonl"

# Store the pre-defined sets of distribution parameters (mean, std_dev).
DISTRIBUTION_SETS = [
  [(12, 4), (10, 6), (8, 4)],
  [(18, 4), (8, 2), (8, 1)],
]

flags.DEFINE_integer(
  name="simulations",
  short_name="s",
  default=10_000,
  lower_bound=1,
  help="The total number of simulations to run.",
)
flags.DEFINE_integer(
  name="periods",
  short_name="p",
  default=300,
  lower_bound=1,
  help="The number of time steps (or trials) in each simulation.",
)
flags.DEFINE_enum(
  name="model",
  short_name="m",
  default="neural_bandit",
  enum_values=[
    "explore_exploit",
    "epsilon_greedy",
    "stable_exploration",
    "neural_bandit",
  ],
  help="The bandit algorithm to use for the simulation.",
)
flags.DEFINE_float(
  name="epsilon",
  short_name="e",
  default=0.1,
  lower_bound=0.0,
  upper_bound=1.0,
  help="Epsilon value for the EpsilonGreedy algorithm.",
)
flags.DEFINE_integer(
  name="distributions",
  short_name="d",
  default=0,
  lower_bound=0,
  upper_bound=len(DISTRIBUTION_SETS) - 1,
  help="The index of the pre-defined distribution set to use (0, 1, ...).",
)
flags.DEFINE_float(
  name="stability_threshold",
  default=0.5,
  lower_bound=0.0,
  help="Std dev change threshold for the StableExploration algorithm.",
)
flags.DEFINE_integer(
  name="consecutive_count",
  default=3,
  lower_bound=1,
  help="Consecutive pulls needed to stabilize for StableExploration.",
)
flags.DEFINE_string(
  name="weights_path",
  default="model/bandit_nn_final.weights.h5",
  help="Path to the trained model weights for the NeuralBandit algorithm.",
)


def main(argv) -> None:
  """Sets up, runs, and logs the Monte Carlo simulation."""
  del argv  # Unused.

  # 1. Select and validate the distribution set.
  set_index = FLAGS.distributions
  selected_params = DISTRIBUTION_SETS[set_index]
  distributions = [(norm.rvs, params) for params in selected_params]
  num_options = len(distributions)

  # 2. Instantiate the chosen model based on the flag.
  config = {}
  if FLAGS.model == "explore_exploit":
    model = explore.ExploreExploit(num_options=num_options)
    config["algorithm"] = "Explore Exploit"
  elif FLAGS.model == "epsilon_greedy":
    model = epsilon.EpsilonGreedy(
      num_options=num_options, epsilon=FLAGS.epsilon
    )
    config["algorithm"] = "Epsilon Greedy"
    config["epsilon"] = FLAGS.epsilon
  elif FLAGS.model == "stable_exploration":
    model = stable.StableExploration(
      num_options=num_options,
      stability_threshold=FLAGS.stability_threshold,
      consecutive_count_needed=FLAGS.consecutive_count,
    )
    config["algorithm"] = "Stable Exploration"
    config["stability"] = FLAGS.stability_threshold
    config["consecutive"] = FLAGS.consecutive_count
  elif FLAGS.model == "neural_bandit":
    model = neural.NeuralBandit(
      num_options=num_options,
      total_periods=FLAGS.periods,  # The NN needs to know the horizon
      weights_path=FLAGS.weights_path,
    )
    config["algorithm"] = "Neural Bandit"
    config["weights_path"] = FLAGS.weights_path
  else:
    raise ValueError(f"Unknown model specified: {FLAGS.model}")

  dist_str = ", ".join([f"N({m},{s})" for m, s in selected_params])
  config["distributions"] = f"{set_index} [{dist_str}]"
  config["periods"] = FLAGS.periods
  config["simulations"] = FLAGS.simulations

  print("--- Configuration ---")
  for key, value in config.items():
    print(f"{key:<20}{value}")

  # 3. Set up and run the simulation.
  print("\n--- Running ---")
  sim_runner = simulation.MonteCarlo(
    model=model,
    distributions=distributions,
    periods=FLAGS.periods,
    num_simulations=FLAGS.simulations,
  )
  avg, std_dev = sim_runner.run()

  print("\n--- Completed ---")
  print(f"{'Average:':<20}{avg:.2f}")
  print(f"{'Std Dev:':<20}{std_dev:.2f}\n")

  # 4. Log the results to a file.
  log_results(
    model_name=FLAGS.model,
    avg_reward=avg,
    std_dev=std_dev,
    dist_set_index=set_index,
  )
  print(f"Results appended to {RESULTS_FILE}")


def log_results(
  model_name: str, avg_reward: float, std_dev: float, dist_set_index: int
) -> None:
  """Appends the results of a simulation run to a JSON Lines file."""
  result_data = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "algorithm": model_name,
    "parameters": {},
    "configuration": {
      "periods": FLAGS.periods,
      "num_simulations": FLAGS.simulations,
      "distribution_set": dist_set_index,
      "distributions": DISTRIBUTION_SETS[dist_set_index],
    },
    "results": {"mean_reward": avg_reward, "std_dev": std_dev},
  }

  if model_name == "epsilon_greedy":
    result_data["parameters"]["epsilon"] = FLAGS.epsilon
  elif model_name == "stable_exploration":
    result_data["parameters"]["stability_threshold"] = FLAGS.stability_threshold
    result_data["parameters"]["consecutive_count"] = FLAGS.consecutive_count
  elif model_name == "neural_bandit":
    result_data["parameters"]["weights_path"] = FLAGS.weights_path

  with open(RESULTS_FILE, "a") as f:
    f.write(json.dumps(result_data) + "\n")


if __name__ == "__main__":
  app.run(main)
