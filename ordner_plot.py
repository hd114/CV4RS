import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os

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

def extract_benchmark(file_path, max_rounds=40):
    """
    Extract the first `max_rounds` values from a benchmark file.

    Args:
        file_path (str): Path to the benchmark file.
        max_rounds (int): Maximum number of values to extract.

    Returns:
        list: Extracted benchmark values.
    """
    benchmark_values = []
    pattern = r"^\d+\.\d+,\s*(\d+\.\d+)"  # Matches: 1.00, 0.5197

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                benchmark_values.append(float(match.group(1)))
            if len(benchmark_values) >= max_rounds:
                break

    return benchmark_values

def average_micro_mAP_from_folder(folder_path):
    """
    Compute the average micro mAP across all .out files in a folder.

    Args:
        folder_path (str): Path to the folder containing .out files.

    Returns:
        list: Averaged micro mAP values across all files.
    """
    all_micro_mAP = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".out"):  # Process only .out files
            file_path = os.path.join(folder_path, file_name)
            micro_mAP_values = extract_micro_mAP(file_path)
            if micro_mAP_values:
                all_micro_mAP.append(micro_mAP_values)

    if not all_micro_mAP:
        raise ValueError(f"No valid .out files found in {folder_path}")

    # Ensure all lists have the same length by truncating to the shortest
    min_length = min(len(lst) for lst in all_micro_mAP)
    all_micro_mAP = [lst[:min_length] for lst in all_micro_mAP]

    # Compute the average across files
    avg_micro_mAP = np.mean(all_micro_mAP, axis=0)
    
    return avg_micro_mAP.tolist()

def plot_micro_mAP(benchmark, avg_mAP_1, avg_mAP_2, folder1_name, folder2_name, output_path):
    """
    Plot the micro mAP values and save the plot to a file.

    Args:
        benchmark (list): Benchmark values.
        avg_mAP_1 (list): Averaged micro mAP from first folder.
        avg_mAP_2 (list): Averaged micro mAP from second folder.
        folder1_name (str): Name of the first folder (used in legend).
        folder2_name (str): Name of the second folder (used in legend).
        output_path (str): Path to save the plot.
    """
    x_values = np.arange(1, len(benchmark) + 1)

    plt.figure(figsize=(16, 9))

    # Plot each list
    plt.plot(x_values, benchmark, marker='d', linestyle='-.', color='grey', linewidth=2, label="Scenario 2 Benchmark")
    plt.plot(x_values, avg_mAP_1, marker='o', linestyle='-', color='green', linewidth=2, label=f"Avg {folder1_name}")
    plt.plot(x_values, avg_mAP_2, marker='s', linestyle='-', color='orange', linewidth=2, label=f"Avg {folder2_name}")

    # Vertical line for pruning round
    pruning_round = 4
    plt.axvline(x=pruning_round, color='gray', linestyle='-.', linewidth=2, label="Pruning Round 4")

    # Labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and axis formatting
    plt.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1, 0.23))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot micro mAP values from log files in two folders.")
    parser.add_argument("folder1", type=str, help="Path to the first folder containing .out files.")
    parser.add_argument("folder2", type=str, help="Path to the second folder containing .out files.")
    args = parser.parse_args()

    # Benchmark file
    benchmark_file = "fed_avg_benchmark_scen1.txt"

    # Extract benchmark values (max 40 rounds)
    benchmark = extract_benchmark(benchmark_file, max_rounds=40)

    # Compute averages from the given folders
    avg_mAP_1 = average_micro_mAP_from_folder(args.folder1)
    avg_mAP_2 = average_micro_mAP_from_folder(args.folder2)

    # Folder names for legend
    folder1_name = os.path.basename(os.path.normpath(args.folder1))
    folder2_name = os.path.basename(os.path.normpath(args.folder2))

    print(f"Averaged micro mAP for {folder1_name}:", avg_mAP_1)
    print(f"Averaged micro mAP for {folder2_name}:", avg_mAP_2)
    print(f"Benchmark values:", benchmark)

    # Plot and save
    output_plot_path = "micro_mAP_plot_final_300dpi.png"
    plot_micro_mAP(benchmark, avg_mAP_1, avg_mAP_2, folder1_name, folder2_name, output_path=output_plot_path)

    print(f"Plot saved to {output_plot_path}")
