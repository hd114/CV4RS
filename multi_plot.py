import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import make_interp_spline

def extract_micro_mAP(file_path):
    """
    Extract micro mAP values from a log file.

    Args:
        file_path (str): Path to the log file.

    Returns:
        list: A list of extracted micro mAP values.
    """
    micro_mAP_values = []
    pattern = r"micro\s+precision:.*?mAP:\s+(\d+\.\d+)"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                micro_mAP_values.append(float(match.group(1)))

    return micro_mAP_values

def smooth_curve(x, y, points=300):
    """
    Smooth the curve using spline interpolation.

    Args:
        x (array): Original x values.
        y (array): Original y values.
        points (int): Number of points for the smoothed curve.

    Returns:
        tuple: Smoothed x and y values.
    """
    spline = make_interp_spline(x, y, k=3)  # Cubic spline
    x_smooth = np.linspace(x.min(), x.max(), points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def plot_micro_mAP(micro_mAP_data, labels, output_path):
    """
    Plot the micro mAP values with a shaded area representing the range and a line plot for the average.

    Args:
        micro_mAP_data (list of lists): List of micro mAP values for each scenario.
        labels (list): List of labels for each scenario.
        output_path (str): Path to save the plot.
    """
    x_values = np.arange(1, len(micro_mAP_data[0]) + 1)

    plt.figure(figsize=(16, 9))

    # Separate data for range calculation
    benchmark_data = micro_mAP_data[1]  # Scenario 2 Benchmark
    other_data = [data for i, data in enumerate(micro_mAP_data) if i != 1]

    # Convert other data to numpy array for range and average calculation
    other_data_array = np.array(other_data)
    mean_values = np.mean(other_data_array, axis=0)
    min_values = np.min(other_data_array, axis=0)
    max_values = np.max(other_data_array, axis=0)

    # Smooth curves for average and range
    x_smooth, mean_smooth = smooth_curve(x_values, mean_values)
    _, min_smooth = smooth_curve(x_values, min_values)
    _, max_smooth = smooth_curve(x_values, max_values)

    # Output the data points for the average line
    print("Average line data points (x, y):")
    for x_val, y_val in zip(x_smooth, mean_smooth):
        print(f"{x_val:.2f}, {y_val:.4f}")

    # Plot shaded area (min to max range) for the other scenarios
    plt.fill_between(x_smooth, min_smooth, max_smooth, color='lightblue', alpha=0.5, label="Range (min-max)")

    # Plot mean line for the other scenarios
    plt.plot(x_smooth, mean_smooth, color='blue', linewidth=2, label="Average micro mAP Fed-Avg (no pruning)")

    # Plot Scenario 2 Benchmark separately (currently commented out)
    # plt.plot(x_values, benchmark_data, linestyle='--', color='red', label="Scenario 2 Benchmark", linewidth=2)

    # Set labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and axis formatting
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot micro mAP values from multiple log files.")
    parser.add_argument("files", nargs='+', type=str, help="Paths to the log files.")
    parser.add_argument("--labels", nargs='+', type=str, default=None, help="Labels for each scenario.")
    parser.add_argument("--scenario2_benchmark", type=str, default="presentation_plots/pytorch_job_2917_country_per_client.out", help="Path to the Scenario 2 Benchmark file.")
    args = parser.parse_args()

    # Extract micro mAP values for each file
    micro_mAP_data = []
    for file in args.files:
        micro_mAP_data.append(extract_micro_mAP(file))

    # Add Scenario 2 Benchmark
    scenario2benchmark = extract_micro_mAP(args.scenario2_benchmark)
    print(f"Scenario 2 Benchmark Data: {scenario2benchmark}")  # Debugging output
    micro_mAP_data.insert(1, scenario2benchmark)

    # Determine the minimum length of completed rounds
    min_length = min(len(data) for data in micro_mAP_data if len(data) > 0)

    # Truncate all data to the length of the shortest file's completed rounds
    micro_mAP_data = [data[:min_length] for data in micro_mAP_data]

    # Labels
    if args.labels is None:
        labels = [f"Scenario {i+1}" for i in range(len(micro_mAP_data))]
    else:
        labels = args.labels

    # Ensure Scenario 2 Benchmark label
    if len(labels) > 1:
        labels[1] = "Scenario 2 Benchmark"

    print(f"Labels: {labels}")  # Debugging output

    # Plot and save
    output_plot_path = "micro_mAP_multi_.png"
    plot_micro_mAP(micro_mAP_data, labels=labels, output_path=output_plot_path)

    print(f"Plot saved to {output_plot_path}")
