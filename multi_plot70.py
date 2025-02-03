import os
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
    
    with open(file_path, 'r') as f:
        for line in f:
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

    # Use all provided datasets for range and average calculation
    data_array = np.array(micro_mAP_data)
    mean_values = np.mean(data_array, axis=0)
    min_values = np.min(data_array, axis=0)
    max_values = np.max(data_array, axis=0)

    # Smooth curves for average and range
    x_smooth, mean_smooth = smooth_curve(x_values, mean_values)
    _, min_smooth = smooth_curve(x_values, min_values)
    _, max_smooth = smooth_curve(x_values, max_values)

    # Output the data points for the average line
    print("Average line data points (x, y):")
    for x_val, y_val in zip(x_smooth, mean_smooth):
        print(f"{x_val:.2f}, {y_val:.4f}")

    # Plot shaded area (min to max range) for the scenarios
    plt.fill_between(x_smooth, min_smooth, max_smooth, color='lightblue', alpha=0.5, label="Range (min-max)")

    # Plot mean line for the scenarios
    plt.plot(x_smooth, mean_smooth, color='blue', linewidth=2, label="Average micro mAP Fed-Avg (no pruning)")

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
    parser = argparse.ArgumentParser(description="Plot micro mAP values from multiple log files.")
    parser.add_argument("inputs", nargs='+', type=str,
                        help="Paths to the log files and optionally labels if they are not valid file paths.")
    parser.add_argument("--labels", nargs='+', type=str, default=None,
                        help="Labels for each scenario (overrides positional labels if provided).")
    args = parser.parse_args()

    # Separate input arguments into valid file paths and positional labels (if any)
    file_list = []
    pos_labels = []
    for inp in args.inputs:
        if os.path.isfile(inp):
            file_list.append(inp)
        else:
            pos_labels.append(inp)

    if not file_list:
        print("No valid input files provided. Exiting.")
        exit(1)

    # Determine labels: if --labels is provided, use these;
    # else, if the number of non-file inputs equals the number of files, use them;
    # otherwise, use default labels.
    if args.labels is not None:
        labels = args.labels
        if len(labels) != len(file_list):
            print("Number of labels provided does not match number of input files. Using default labels.")
            labels = [f"Scenario {i+1}" for i in range(len(file_list))]
    elif len(pos_labels) == len(file_list):
        labels = pos_labels
    else:
        labels = [f"Scenario {i+1}" for i in range(len(file_list))]

    # Extract micro mAP values for each input file
    micro_mAP_data = []
    for file in file_list:
        data = extract_micro_mAP(file)
        micro_mAP_data.append(data)

    # Desired number of rounds to plot
    desired_rounds = 70

    # Filter out datasets that do not have the required number of rounds
    filtered_micro_mAP_data = []
    filtered_labels = []
    for data, label in zip(micro_mAP_data, labels):
        if len(data) >= desired_rounds:
            filtered_micro_mAP_data.append(data[:desired_rounds])
            filtered_labels.append(label)
        else:
            print(f"Skipping dataset '{label}' because it has only {len(data)} rounds.")

    if not filtered_micro_mAP_data:
        print("No dataset has the required number of rounds (70 rounds). Exiting.")
        exit(1)

    output_plot_path = "micro_mAP_multi_.png"
    plot_micro_mAP(filtered_micro_mAP_data, labels=filtered_labels, output_path=output_plot_path)
    print(f"Plot saved to {output_plot_path}")
