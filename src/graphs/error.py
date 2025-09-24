import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_convergence_error(
  ax: plt.Axes, mean: float, std_dev: float, num_samples: int
):
  """
  Generates samples and plots the absolute error of the running mean and
  running standard deviation as they converge to their true values.

  Args:
    ax: The Matplotlib Axes object to draw the plot on.
    mean: The true mean of the normal distribution.
    std_dev: The true standard deviation of the normal distribution.
    num_samples: The number of samples to generate and plot.
  """
  # 1. Generate random samples from the distribution.
  samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
  x_axis = np.arange(1, num_samples + 1)

  # 2. Calculate the running mean.
  running_mean = np.cumsum(samples) / x_axis

  # 3. Calculate the running standard deviation using an efficient one-pass method.
  with np.errstate(divide="ignore", invalid="ignore"):
    running_sum_sq = np.cumsum(samples**2)
    running_sum = np.cumsum(samples)
    # Formula for sample variance
    running_variance = (running_sum_sq - (running_sum**2) / x_axis) / (
      x_axis - 1
    )

  running_variance[0] = 0  # Std dev of a single point is 0
  running_std = np.sqrt(running_variance)

  # 4. Calculate the absolute error for both statistics at each step.
  mean_error = np.abs(running_mean - mean)
  std_error = np.abs(running_std - std_dev)

  # 5. Plot the two error lines on the provided subplot axis.
  sns.lineplot(x=x_axis, y=mean_error, ax=ax, label="Mean Absolute Error")
  sns.lineplot(x=x_axis, y=std_error, ax=ax, label="Std Dev Absolute Error")

  # 6. Set titles and labels for the specific subplot.
  ax.set_title(f"Convergence Error for N(mean={mean}, std={std_dev})")
  ax.set_ylabel("Absolute Error")
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

  # Create a figure with 3 vertically stacked subplots.
  fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), sharex=True)
  sns.set_theme(style="darkgrid")
  fig.suptitle(
    "Convergence Error of Running Mean and Standard Deviation", fontsize=20
  )

  # Generate and draw a plot for each distribution on its subplot.
  for i, (mean, std) in enumerate(distributions):
    plot_convergence_error(
      ax=axes[i],
      mean=mean,
      std_dev=std,
      num_samples=300,
    )

  # Set a single x-axis label for the shared axis.
  axes[-1].set_xlabel("Number of Samples")

  # Adjust layout and save the single figure.
  plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for suptitle

  output_filename = os.path.join(IMG_DIR, "all_distributions_error.png")
  plt.savefig(output_filename)
  plt.close()

  print(f"Error plot saved as {output_filename}")
