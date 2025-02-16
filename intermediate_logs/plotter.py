import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def plot_micro_mAP(scenario1benchmark, scenario2benchmark, scenario2lrp03, scenario2lrp06, output_path):
    """
    Plot the micro mAP values and save the plot to a file.

    Args:
        scenario1benchmark (list): Values for Scenario 1 Benchmark.
        scenario2benchmark (list): Values for Scenario 2 Benchmark.
        scenario2lrp03 (list): Values for Scenario 2 LRP-pruning with rate 0.3.
        scenario2lrp06 (list): Values for Scenario 2 LRP-pruning with rate 0.6.
        output_path (str): Path to save the plot.
    """
    x_values = np.arange(1, len(scenario1benchmark) + 1)

    plt.figure(figsize=(16, 9))

    # Plot each list
    #plt.plot(x_values, scenario1benchmark, marker='^', linestyle='--', color='green', linewidth=2, label="Scenario 1 Benchmark")
    plt.plot(x_values, scenario2benchmark, marker='d', linestyle='-.', color='red', linewidth=2, label="Scenario 2 Benchmark")
    plt.plot(x_values, scenario2lrp03, marker='o', linestyle='-', color='blue', linewidth=2, label="Scenario 2 LRP-pruning with rate 0.3")
    plt.plot(x_values, scenario2lrp06, marker='s', linestyle='-', color='purple', linewidth=2, label="Scenario 2 LRP-pruning with rate 0.6")

    # Vertical line for pruning round
    pruning_round = 4
    plt.axvline(x=pruning_round, color='gray', linestyle='--', linewidth=2, label="Pruning Round 4")

    # Labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and axis formatting
    plt.legend(fontsize=20, loc='lower right', bbox_to_anchor=(1, 0.23))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # File paths
    file_scenario1_benchmark = "pytorch_job_3723_scenario1.out"
    file_scenario2_benchmark = "pytorch_job_2917_country_per_client.out"
    file_scenario2lrp03 = "alldata_0.3prun_40rnds_1x8ctr.out"
    file_scenario2lrp06 = "alldata_0.6prun_40rnds_1x8ctr.out"

    # Extract micro mAP values
    scenario1benchmark = extract_micro_mAP(file_scenario1_benchmark)
    scenario2benchmark = extract_micro_mAP(file_scenario2_benchmark)
    scenario2lrp03 = extract_micro_mAP(file_scenario2lrp03)
    scenario2lrp06 = extract_micro_mAP(file_scenario2lrp06)

    print("Scenario 1 Benchmark:", scenario1benchmark)
    print("Scenario 2 Benchmark:", scenario2benchmark)
    print("Scenario 2 LRP 0.3:", scenario2lrp03)
    print("Scenario 2 LRP 0.6:", scenario2lrp06)
    
    # Plot and save
    output_plot_path = "micro_mAP_plot_nogreen_300dpi.png"
    plot_micro_mAP(scenario1benchmark, scenario2benchmark, scenario2lrp03, scenario2lrp06, output_path=output_plot_path)

    print(f"Plot saved to {output_plot_path}")

