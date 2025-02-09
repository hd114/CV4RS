import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def plot_micro_mAP_per_rate(to_plot):
    """
        benchmark,three,five,seven and ninetyfive all are lists of len 3. Each contain name,max value, value list
        Plot the micro mAP values and save the plot in high resolution.
    """
    benchmark, three, five, seven, ninetyfive = to_plot

    scenario = benchmark[0][0]
    strategy = three[0].split(" ")[1]

    x_values = np.arange(1, len(benchmark[2]) + 1)

    plt.figure(figsize=(16, 9))

    plt.plot(x_values, benchmark[2], marker=None, linestyle='solid', color='grey', linewidth=2, label=f"Scenario {scenario} Benchmark")
    plt.plot(x_values, three[2], marker='^', linestyle='solid', color='green', linewidth=2, label=f"Scenario {scenario} {strategy}-pruning with rate 0.3") #dashdot
    plt.plot(x_values, five[2], marker='d', linestyle='solid', color='orange', linewidth=2, label=f"Scenario {scenario} {strategy}-pruning with rate 0.5") #dashed
    plt.plot(x_values, seven[2], marker='o', linestyle='solid', color='blue', linewidth=2, label=f"Scenario {scenario} {strategy}-pruning with rate 0.7") #dashed
    plt.plot(x_values, ninetyfive[2], marker='s', linestyle='solid', color='purple', linewidth=2, label=f"Scenario {scenario} {strategy}-pruning with rate 0.95") #dotted

    # Vertical line for pruning round
    pruning_round = 4
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
    plt.savefig(f"sce{scenario}_{strategy}.png", dpi=300)
    plt.close()

def plot_micro_mAP_per_strat(to_plot):
    """
        benchmark, random, l2, nisp all are lists of len 3. Each contain name,max value, value list
        Plot the micro mAP values and save the plot in high resolution.
    """
    benchmark, random, l2, nisp = to_plot

    scenario = benchmark[0][0]
    rate = random[0].split(" ")[-1]

    x_values = np.arange(1, len(benchmark[2]) + 1)

    plt.figure(figsize=(16, 9))

    plt.plot(x_values, benchmark[2], marker=None, linestyle='solid', color='grey', linewidth=2, label=f"Scenario {scenario} Benchmark")
    plt.plot(x_values, random[2], marker='^', linestyle='solid', color='green', linewidth=2, label=f"Scenario {scenario} Random-pruning with rate {rate}") #dashdot
    plt.plot(x_values, l2[2], marker='d', linestyle='solid', color='orange', linewidth=2, label=f"Scenario {scenario} L2-pruning with rate {rate}") #dashed
    plt.plot(x_values, nisp[2], marker='o', linestyle='solid', color='blue', linewidth=2, label=f"Scenario {scenario} Nisp-pruning with rate {rate}") #dashed
    #plt.plot(x_values, ninetyfive[2], marker='s', linestyle='solid', color='purple', linewidth=2, label=f"Scenario {scenario} {strategy}-pruning with rate 0.95") #dotted

    # Vertical line for pruning round
    pruning_round = 4
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
    plt.savefig(f"sce{scenario}_{rate}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    scenarios = ["1","2"]
    strategies = ["Random","L2","Nisp"]
    pruning_rates = ["0.3","0.5","0.7","0.95"]
    logs_directory="../logs/"

    value_lists_dict = {}
    max_values = {}
    max_value_indices = {}

    for log_file_name in os.listdir(logs_directory):
        log_path = os.path.join(logs_directory, log_file_name)
        if os.path.isfile(log_path):
            log_name = log_file_name.split("_")[0:3]
            if log_name[1]!="unpruned":
                name = " ".join([log_name[0][-1], log_name[1].capitalize(), "0."+str(int(log_name[2][2:]))])
            else:
                name = log_name[0][-1] + " " + "Benchmark"
            value_lists_dict[name] = extract_micro_mAP(log_path)
            max_values[name] = max(value_lists_dict[name])
            max_value_indices[name] = value_lists_dict[name].index(max_values[name])


    for scenario in scenarios:
        for strategy in strategies:
            to_plot = [[scenario+" Benchmark",max_value_indices[scenario+" Benchmark"],value_lists_dict[scenario+" Benchmark"]]]
            for rate in pruning_rates:
                name = scenario + " " + strategy + " " + rate
                to_plot.append([name,max_value_indices[name],value_lists_dict[name]])
            plot_micro_mAP_per_rate(to_plot)

        for rate in pruning_rates:
            to_plot = [[scenario+" Benchmark",max_value_indices[scenario+" Benchmark"],value_lists_dict[scenario+" Benchmark"]]]
            for strategy in strategies:
                name = scenario + " " + strategy + " " + rate
                to_plot.append([name, max_value_indices[name], value_lists_dict[name]])
            plot_micro_mAP_per_strat(to_plot)

if __name__ == "o__main__":
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
    output_plot_path = "micro_mAP_plot_final_300dpi.png"
    plot_micro_mAP(scenario1benchmark, scenario2benchmark, scenario2lrp03, scenario2lrp06, output_path=output_plot_path)

    print(f"Plot saved to {output_plot_path}")

