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

def plot_micro_mAP(micro_mAP_data, labels, output_path, benchmark1=None, benchmark2=None):
    """
    Plot the micro mAP values with a shaded area representing the range and a line plot for the average.
    Additionally, plot benchmark curves if provided.

    Args:
        micro_mAP_data (list of lists): List of micro mAP values for each scenario (input files).
        labels (list): List of labels for each scenario.
        output_path (str): Path to save the plot.
        benchmark1 (list, optional): Benchmark values for Scenario 1.
        benchmark2 (list, optional): Benchmark values for Scenario 2.
    """
    # x_values: 1 bis Anzahl der Runden (z. B. 1 bis 40)
    x_values = np.arange(1, len(micro_mAP_data[0]) + 1)

    plt.figure(figsize=(16, 9))

    # Compute average, min, and max over the input files
    data_array = np.array(micro_mAP_data)
    mean_values = np.mean(data_array, axis=0)
    min_values = np.min(data_array, axis=0)
    max_values = np.max(data_array, axis=0)

    # Smooth curves for average and range (for plotting)
    x_smooth, mean_smooth = smooth_curve(x_values, mean_values)
    _, min_smooth = smooth_curve(x_values, min_values)
    _, max_smooth = smooth_curve(x_values, max_values)

    # Ausgabe der unsmoothten Durchschnittswerte (1 Datenpunkt pro Runde)
    print("Average line data points (x, y):")
    for x_val, y_val in zip(x_values, mean_values):
        print(f"{x_val:.2f}, {y_val:.4f}")

    # Plot shaded area (min to max range)
    plt.fill_between(x_smooth, min_smooth, max_smooth, color='lightblue', alpha=0.5, label="Range (min-max)")

    # Plot average line for input data
    plt.plot(x_smooth, mean_smooth, color='blue', linewidth=2, label="Average micro mAP LRP (0.97 / 0.04)")

    # Plot benchmark curves if available
    if benchmark1 is not None:
        # Benchmark1: rote, gestrichelte Linie
        benchmark1 = benchmark1[:len(x_values)]
        plt.plot(x_values, benchmark1, linestyle='--', color='red', linewidth=2, label="Scenario 1 Benchmark")
    if benchmark2 is not None:
        # Benchmark2: magentafarbene, gestrichelte Linie
        benchmark2 = benchmark2[:len(x_values)]
        plt.plot(x_values, benchmark2, linestyle='--', color='magenta', linewidth=2, label="Scenario 2 Benchmark")

    # Labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and tick formatting
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot micro mAP values from multiple log files.")
    parser.add_argument("files", nargs='+', type=str,
                        help="Paths to the input log files.")
    parser.add_argument("--labels", nargs='+', type=str, default=None,
                        help="Labels for each input scenario (overrides positional labels if provided).")
    parser.add_argument("--benchmark1", type=str, default="fed_avg_benchmark_scen1.txt",
                        help="Path to the Scenario 1 Benchmark file.")
    parser.add_argument("--benchmark2", type=str, default="fed_avg_benchmark_scen2.txt",
                        help="Path to the Scenario 2 Benchmark file.")
    args = parser.parse_args()

    # Extract micro mAP values from input files
    micro_mAP_data = []
    for file in args.files:
        micro_mAP_data.append(extract_micro_mAP(file))

    # Standard: gewünschte Anzahl an Runden = 40
    desired_rounds = 40
    filtered_micro_mAP_data = []
    # Default-Labels, falls keine --labels angegeben werden
    default_labels = [f"Scenario {i+1}" for i in range(len(micro_mAP_data))]
    input_labels = args.labels if args.labels is not None else default_labels
    for data, label in zip(micro_mAP_data, input_labels):
        if len(data) >= desired_rounds:
            filtered_micro_mAP_data.append(data[:desired_rounds])
        else:
            print(f"Skipping dataset '{label}' because it has only {len(data)} rounds.")

    if not filtered_micro_mAP_data:
        print("No dataset has the required number of rounds (40 rounds). Exiting.")
        exit(1)

    # Extract benchmark data from benchmark files
    benchmark1_data = extract_micro_mAP(args.benchmark1)
    if len(benchmark1_data) < desired_rounds:
        print(f"Warning: Benchmark file {args.benchmark1} has less than {desired_rounds} rounds; skipping Scenario 1 Benchmark.")
        benchmark1_data = None
    else:
        benchmark1_data = benchmark1_data[:desired_rounds]

    benchmark2_data = extract_micro_mAP(args.benchmark2)
    if len(benchmark2_data) < desired_rounds:
        print(f"Warning: Benchmark file {args.benchmark2} has less than {desired_rounds} rounds; skipping Scenario 2 Benchmark.")
        benchmark2_data = None
    else:
        benchmark2_data = benchmark2_data[:desired_rounds]

    print(f"Input Labels: {input_labels}")  # Debugging output

    output_plot_path = "micro_mAP_multi_.png"
    plot_micro_mAP(filtered_micro_mAP_data, labels=input_labels, output_path=output_plot_path,
                   benchmark1=benchmark1_data, benchmark2=benchmark2_data)

    print(f"Plot saved to {output_plot_path}")
