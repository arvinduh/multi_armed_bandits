import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_distribution_convergence(
  ax: plt.Axes, mean: float, std_dev: float, num_samples: int
):
  """
  Generates samples from a normal distribution and plots the convergence
  of the running mean and running standard deviation on a given subplot axis.

  Args:
    ax: The Matplotlib Axes object to draw the plot on.
    mean: The true mean of the normal distribution.
    std_dev: The true standard deviation of the normal distribution.
    num_samples: The number of samples to generate and plot.
  """
  # 1. Generate random samples.
  samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
  x_axis = np.arange(1, num_samples + 1)

  # 2. Calculate the running mean efficiently.
  running_mean = np.cumsum(samples) / x_axis

  # 3. Calculate the running standard deviation efficiently.
  # This uses a vectorized one-pass algorithm to avoid slow loops.
  with np.errstate(divide="ignore", invalid="ignore"):
    # Formula for sample variance: (E[X^2] - (E[X])^2) * (n / (n-1))
    running_sum = np.cumsum(samples)
    running_sum_sq = np.cumsum(samples**2)
    running_variance = (running_sum_sq - (running_sum**2) / x_axis) / (
      x_axis - 1
    )

  # The standard deviation of a single point is 0.
  running_variance[0] = 0
  running_std = np.sqrt(running_variance)

  # 4. Define the upper and lower bounds using the running std dev.
  upper_bound = running_mean + running_std
  lower_bound = running_mean - running_std

  # 5. Plot all components on the provided subplot axis (ax).
  sns.lineplot(x=x_axis, y=running_mean, ax=ax, label="Running Mean")
  ax.fill_between(
    x_axis, lower_bound, upper_bound, alpha=0.2, label="Running Std Dev"
  )
  ax.axhline(y=mean, color="r", linestyle="--", label=f"True Mean ({mean})")

  # 6. Set titles and labels for the specific subplot.
  ax.set_title(f"Convergence for N(mean={mean}, std={std_dev})")
  ax.set_ylabel("Value")
  ax.legend()
  ax.grid(True)


if __name__ == "__main__":
  # Define the directory to save images and create it if it doesn't exist.
  IMG_DIR = "img"
  os.makedirs(IMG_DIR, exist_ok=True)

  # Define the sets of distributions (mean, std_dev) to plot.
  distributions = [
    (12, 4),  # N(12, 4)
    (10, 6),  # N(10, 6)
    (8, 4),  # N(8, 4)
  ]

  # 1. Create a figure with 3 vertically stacked subplots.
  # `sharex=True` links the x-axes for a cleaner look.
  fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)
  sns.set_theme(style="darkgrid")
  fig.suptitle(
    "Convergence of Running Mean and Standard Deviation", fontsize=20
  )

  # 2. Generate and draw a plot for each distribution on its respective subplot.
  for i, (mean, std) in enumerate(distributions):
    plot_distribution_convergence(
      ax=axes[i],
      mean=mean,
      std_dev=std,
      num_samples=300,
    )

  # 3. Set a single x-axis label for the shared axis.
  axes[-1].set_xlabel("Number of Samples")

  # 4. Adjust layout and save the single figure.
  plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for suptitle

  output_filename = os.path.join(IMG_DIR, "convergence.png")
  plt.savefig(output_filename)
  plt.close()

  print(f"Combined plot saved as {output_filename}")
