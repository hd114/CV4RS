import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

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

def plot_micro_mAP(scenario1benchmark, scenario2benchmark, scenario2lrp03, scenario2lrp06, output_path, max_rounds):
    """
    Plot the micro mAP values and save the plot to a file.

    Args:
        scenario1benchmark (list): Values for Scenario 1 Benchmark.
        scenario2benchmark (list): Values for Scenario 2 Benchmark.
        scenario2lrp03 (list): Values for Scenario 2 LRP-pruning with rate 0.3.
        scenario2lrp06 (list): Values for Scenario 2 LRP-pruning with rate 0.6.
        output_path (str): Path to save the plot.
        max_rounds (int): Maximum number of communication rounds to plot.
    """
    x_values = np.arange(1, max_rounds + 1)

    plt.figure(figsize=(16, 9))

    # Plot each list, truncated to max_rounds
    if len(scenario1benchmark) >= max_rounds:
        plt.plot(x_values, scenario1benchmark[:max_rounds], marker='^', linestyle='--', color='green', linewidth=2, label="Scenario 1 Benchmark")
    if len(scenario2benchmark) >= max_rounds:
        plt.plot(x_values, scenario2benchmark[:max_rounds], marker='d', linestyle='-.', color='red', linewidth=2, label="Scenario 2 Benchmark")
    if len(scenario2lrp03) >= max_rounds:
        plt.plot(x_values, scenario2lrp03[:max_rounds], marker='o', linestyle='-', color='blue', linewidth=2, label="Scenario 1 LRP 0.97 prun, 0.04 retain, egge")
    if len(scenario2lrp06) >= max_rounds:
        plt.plot(x_values, scenario2lrp06[:max_rounds], marker='s', linestyle='-', color='purple', linewidth=2, label="Scenario 2 LRP-pruning with rate 0.6")

    # Vertical line for pruning round (optional, can be adjusted)
    pruning_round = min(4, max_rounds)
    plt.axvline(x=pruning_round, color='gray', linestyle='--', linewidth=2, label="Pruning Round 4")

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
    parser = argparse.ArgumentParser(description="Plot micro mAP values from log files.")
    parser.add_argument("file_scenario2lrp03", type=str, help="Path to the file for Scenario 2 LRP-pruning with rate 0.3.")
    parser.add_argument("--max_rounds", type=int, default=50, help="Maximum number of communication rounds to plot.")
    args = parser.parse_args()

    # File paths
    file_scenario1_benchmark = "presentation_plots/pytorch_job_3723_scenario1.out"
    file_scenario2_benchmark = "presentation_plots/pytorch_job_2917_country_per_client.out"
    file_scenario2lrp06 = "presentation_plots/alldata_0.6prun_40rnds_1x8ctr.out"

    # Extract micro mAP values
    scenario1benchmark = extract_micro_mAP(file_scenario1_benchmark)
    scenario2benchmark = extract_micro_mAP(file_scenario2_benchmark)
    scenario2lrp03 = extract_micro_mAP(args.file_scenario2lrp03)
    scenario2lrp06 = extract_micro_mAP(file_scenario2lrp06)

    print("Scenario 1 Benchmark:", scenario1benchmark)
    print("Scenario 2 Benchmark:", scenario2benchmark)
    print("Scenario 2 LRP 0.98, 0.03 retain:", scenario2lrp03)
    print("Scenario 2 LRP 0.6:", scenario2lrp06)

    # Plot and save
    output_plot_path = "micro_mAP_plot_final_300dpi.png"
    plot_micro_mAP(scenario1benchmark, scenario2benchmark, scenario2lrp03, scenario2lrp06, output_path=output_plot_path, max_rounds=args.max_rounds)

    print(f"Plot saved to {output_plot_path}")
