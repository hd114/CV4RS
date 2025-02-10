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

    return np.array(micro_mAP_values)


def plot_micro_mAP_per_rate(to_plot):
    """
        benchmark,three,five,seven and ninetyfive all are lists of len 3. Each contain name,max value, value list
        Plot the micro mAP values and save the plot in high resolution.
    """
    benchmark, three, five, seven, ninetyfive = to_plot

    scenario = benchmark[0][0]
    strategy = three[0] #split(" ")[1]

    x_values = np.arange(1, len(benchmark[2]) + 1)

    plt.figure(figsize=(16, 9))

    plt.plot(x_values, benchmark[2], marker=None, linestyle='solid', color='grey', linewidth=2, label=f"Scenario {scenario} Benchmark")
    plt.plot(x_values, three[2], marker='^', linestyle='solid', color='green', linewidth=2, label=f"Scenario {scenario} {strategy} with rate 0.3")  # dashdot
    plt.plot(x_values, five[2], marker='d', linestyle='solid', color='orange', linewidth=2, label=f"Scenario {scenario} {strategy} with rate 0.5")  # dashed
    plt.plot(x_values, seven[2], marker='o', linestyle='solid', color='blue', linewidth=2, label=f"Scenario {scenario} {strategy} with rate 0.7")  # dashed
    plt.plot(x_values, ninetyfive[2], marker='s', linestyle='solid', color='purple', linewidth=2, label=f"Scenario {scenario} {strategy} with rate 0.95")  # dotted

    # Vertical line for pruning round
    pruning_round = 4
    plt.axvline(x=pruning_round, color='gray', linestyle='--', linewidth=2, label="Pruning Round 4")

    # Labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and axis formatting
    plt.legend(fontsize=25, loc='lower right', bbox_to_anchor=(1, 0.23))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(f"plots/t3sce{scenario}_{strategy}.png", dpi=200)
    plt.close()


def plot_micro_mAP_per_strat(to_plot):
    """
        benchmark, random, nisp #l2,  all are lists of len 3. Each contain name,max value, value list
        Plot the micro mAP values and save the plot in high resolution.
    """
    benchmark, random, nisp = to_plot #l2,

    scenario = benchmark[0][0]
    rate = benchmark[1] #random[0].split(" ")[-1]

    x_values = np.arange(1, len(benchmark[2]) + 1)

    plt.figure(figsize=(16, 9))

    plt.plot(x_values, benchmark[2], marker=None, linestyle='solid', color='grey', linewidth=2, label=f"Scenario {scenario} Benchmark")
    plt.plot(x_values, random[2], marker='^', linestyle='solid', color='green', linewidth=2, label=f"Scenario {scenario} Random-pruning with rate {rate}")  # dashdot label=f"Scenario {scenario} L2-pruning with rate {rate}")  # dashed
    plt.plot(x_values, nisp[2], marker='o', linestyle='solid', color='blue', linewidth=2, label=f"Scenario {scenario} FedNISP with rate {rate}")  # dashed
    # plt.plot(x_values, ninetyfive[2], marker='s', linestyle='solid', color='purple', linewidth=2, label=f"Scenario {scenario} {strategy}-pruning with rate 0.95") #dotted

    # Vertical line for pruning round
    pruning_round = 4
    plt.axvline(x=pruning_round, color='gray', linestyle='--', linewidth=2, label="Pruning Round 4")

    # Labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and axis formatting
    plt.legend(fontsize=25, loc='lower right', bbox_to_anchor=(1, 0.23))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(f"plots/sce{scenario}_{rate}.png", dpi=200)
    plt.close()

def plot_micro_mAP_per_strat_incl_lrp(to_plot):
    """
        benchmark, random, nisp #l2,  all are lists of len 3. Each contain name,max value, value list
        Plot the micro mAP values and save the plot in high resolution.
    """
    benchmark, random, nisp, lrp = to_plot #l2,

    scenario = benchmark[0][0]
    rate = benchmark[1] #random[0].split(" ")[-1]

    x_values = np.arange(1, len(benchmark[2]) + 1)

    plt.figure(figsize=(16, 9))

    plt.plot(x_values, benchmark[2], marker=None, linestyle='solid', color='grey', linewidth=2, label=f"Scenario {scenario} Benchmark")
    plt.plot(x_values, random[2], marker='^', linestyle='solid', color='green', linewidth=2, label=f"Scenario {scenario} Random-pruning with rate {rate}")  # dashdot label=f"Scenario {scenario} L2-pruning with rate {rate}")  # dashed
    plt.plot(x_values, nisp[2], marker='o', linestyle='solid', color='blue', linewidth=2, label=f"Scenario {scenario} FedNISP with rate {rate}")  # dashed
    plt.plot(x_values, lrp[2], marker='s', linestyle='solid', color='purple', linewidth=2, label=f"Scenario {scenario} LRP-pruning with rate {rate}") #dotted

    # Vertical line for pruning round
    pruning_round = 4
    plt.axvline(x=pruning_round, color='gray', linestyle='--', linewidth=2, label="Pruning Round 4")

    # Labels and grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Legend and axis formatting
    plt.legend(fontsize=25, loc='lower right', bbox_to_anchor=(1, 0.23))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(f"plots_incl_lrp/sce{scenario}_{rate}.png", dpi=200)
    plt.close()

def is_log_name(path):
    return path[-4:]==".out"

def is_t3(path):
    return path[:2]=="t3"

def is_r18(path):
    return path[:3]=="r18"

def remove_job_suffix(name_str):
    return name_str[:-9]

if __name__ == "i__main__":
    path_names = os.listdir(".")
    print("len(path_names)==",len(path_names))

    r18_names = list(filter(is_r18,path_names))
    print("len(log_names)==",len(r18_names),"\n\n\n")

    r18_logs = map(remove_job_suffix,r18_names)
    print("\n".join(r18_logs))

    for name in r18_names:
        extracted = extract_micro_mAP(name)
        print(np.max(extracted),"in round:",extracted.argmax()+1)


if __name__ == "__main__":
    path_names = os.listdir(".")
    print("len(path_names)==",len(path_names))

    t3_paths = list(filter(is_t3,path_names))
    print("len(log_names)==",len(t3_paths),"\n\n")

    #t3_logs = sorted(map(remove_job_suffix,t3_names))
    #print("\n".join(sorted(t3_paths)))

    values = {}

    for t3_path in t3_paths:
        t3_name = remove_job_suffix(t3_path)

        t3_extracted = extract_micro_mAP(t3_path)

        assert len(t3_extracted) == 40

        if t3_name in values:
            values[t3_name].append(t3_extracted)
        else:
            values[t3_name] = [t3_extracted]

    means = {}

    for key in sorted(values.keys()):
        #print("\n",key,len(values[key]))
        means[key] = sum(values[key])/len(values[key])
        #print(key, means[key])
        print(key,"\t",means[key].argmax(), max(means[key]) )


    ##################################################################################################
    scenarios = ["1", "2"]
    strategies = ["random", "nisp","lrp"] # "L2",
    pruning_rates = ["0.3","0.95"]#, ,"0.5", "0.7",

    for scenario in scenarios:
        for rate in pruning_rates:
            benchmark = f"t3sce{scenario}_unpruned"
            random = f"t3sce{scenario}_random_pr0{rate[2:]}"
            nisp = f"t3sce{scenario}_nisp_pr0{rate[2:]}"
            lrp = f"t3sce{scenario}_lrp_pr0{rate[2:]}"

            to_plot = (
                [scenario, rate, means[benchmark]],
                ["Random-pruning", -1, means[random]],
                ["FedNISP", -1, means[nisp]],
                ["LRP-pruning", -1, means[lrp]],
            )

            plot_micro_mAP_per_strat_incl_lrp(to_plot)

if __name__ == "__main__":
    path_names = os.listdir(".")
    print("len(path_names)==",len(path_names))

    t3_paths = list(filter(is_t3,path_names))
    print("len(log_names)==",len(t3_paths),"\n\n")

    #t3_logs = sorted(map(remove_job_suffix,t3_names))
    #print("\n".join(sorted(t3_paths)))

    values = {}

    for t3_path in t3_paths:
        t3_name = remove_job_suffix(t3_path)

        t3_extracted = extract_micro_mAP(t3_path)

        assert len(t3_extracted) == 40

        if t3_name in values:
            values[t3_name].append(t3_extracted)
        else:
            values[t3_name] = [t3_extracted]

    means = {}

    for key in sorted(values.keys()):
        #print("\n",key,len(values[key]))
        means[key] = sum(values[key])/len(values[key])
        #print(key, means[key])
        print(key,"\t",means[key].argmax(), max(means[key]) )


    ##################################################################################################
    scenarios = ["1", "2"]
    strategies = ["random", "nisp"] # "L2",
    pruning_rates = ["0.3", "0.5", "0.7", "0.95"]

    for scenario in scenarios:
        for strategy in strategies:
            benchmark = f"t3sce{scenario}_unpruned"
            three = f"t3sce{scenario}_{strategy}_pr03"
            five = f"t3sce{scenario}_{strategy}_pr05"
            seven = f"t3sce{scenario}_{strategy}_pr07"
            ninetyfive = f"t3sce{scenario}_{strategy}_pr095"

            to_plot = (
                [scenario,-1,means[benchmark]],
                [strategy.capitalize(),-1,means[three]],
                [strategy.capitalize(),-1,means[five]],
                [strategy.capitalize(),-1,means[seven]],
                [strategy.capitalize(),-1,means[ninetyfive]]
                )

            plot_micro_mAP_per_rate(to_plot)

        for rate in pruning_rates:
            benchmark = f"t3sce{scenario}_unpruned"
            random = f"t3sce{scenario}_random_pr0{rate[2:]}"
            nisp = f"t3sce{scenario}_nisp_pr0{rate[2:]}"

            to_plot = (
                [scenario, rate, means[benchmark]],
                ["Random-pruning", -1, means[random]],
                ["FedNISP", -1, means[nisp]],
            )

            plot_micro_mAP_per_strat(to_plot)



if __name__ == "o__main__":
    scenarios = ["1", "2"]
    strategies = ["Random", "L2", "Nisp"]
    pruning_rates = ["0.3", "0.5", "0.7", "0.95"]
    logs_directory = "../logs/"

    value_lists_dict = {}
    max_values = {}
    max_value_indices = {}

    for log_file_name in os.listdir(logs_directory):
        log_path = os.path.join(logs_directory, log_file_name)
        if os.path.isfile(log_path):
            log_name = log_file_name.split("_")[0:3]
            if log_name[1] != "unpruned":
                name = " ".join([log_name[0][-1], log_name[1].capitalize(), "0." + str(int(log_name[2][2:]))])
            else:
                name = log_name[0][-1] + " " + "Benchmark"
            value_lists_dict[name] = extract_micro_mAP(log_path)
            max_values[name] = max(value_lists_dict[name])
            print(name, max_values[name])
            max_value_indices[name] = value_lists_dict[name].index(max_values[name])

    for scenario in scenarios:
        for strategy in strategies:
            to_plot = [[scenario + " Benchmark", max_value_indices[scenario + " Benchmark"],
                        value_lists_dict[scenario + " Benchmark"]]]
            for rate in pruning_rates:
                name = scenario + " " + strategy + " " + rate
                to_plot.append([name, max_value_indices[name], value_lists_dict[name]])
            plot_micro_mAP_per_rate(to_plot)

        for rate in pruning_rates:
            to_plot = [[scenario + " Benchmark", max_value_indices[scenario + " Benchmark"],
                        value_lists_dict[scenario + " Benchmark"]]]
            for strategy in strategies:
                name = scenario + " " + strategy + " " + rate
                to_plot.append([name, max_value_indices[name], value_lists_dict[name]])
            plot_micro_mAP_per_strat(to_plot)


