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
    
    If the file starts with a header like "Average line data points (x, y):",
    a regex is used to extract lines in the format "1.00, 0.5197".
    
    Args:
        file_path (str): Path to the log file.
    
    Returns:
        list: A list of extracted micro mAP values.
    """
    micro_mAP_values = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Prüfe, ob die erste Zeile den Header enthält
    if lines and "Average line data points" in lines[0]:
        pattern = r"^\s*\d+\.\d+\s*,\s*(\d+\.\d+)"
    else:
        pattern = r"micro\s+precision:.*?mAP:\s+(\d+\.\d+)"
    
    for line in lines:
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
    spline = make_interp_spline(x, y, k=1)  # Cubic spline
    x_smooth = np.linspace(x.min(), x.max(), points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def plot_micro_mAP(micro_mAP_data, labels, output_path):
    """
    Plot the micro mAP values with a shaded area representing the range and a line plot for the average.
    Falls Benchmark-Daten vorhanden sind, wird der zweite Datensatz als Benchmark (rote, gestrichelte Linie)
    geplottet – hier werden die rohen Daten aus der TXT-Datei genutzt und zusätzlich mit Spline-Glättung (k=3) versehen.
    
    Eine senkrechte, graue, gestrichelte Linie markiert bei Round 4 den Pruningschritt.
    
    Args:
        micro_mAP_data (list of lists): List of micro mAP values for each scenario.
        labels (list): List of labels for each scenario.
        output_path (str): Path to save the plot.
    """
    # x_values: Runden 1 bis Anzahl der Datenpunkte (z. B. 1 bis 40)
    x_values = np.arange(1, len(micro_mAP_data[0]) + 1)
    
    plt.figure(figsize=(16, 9))
    
    # Separiere Benchmark-Daten (zweiter Datensatz) von den "anderen" Szenarien
    if len(micro_mAP_data) > 1:
        benchmark_data = micro_mAP_data[1]  # Benchmark-Daten aus fed_avg_benchmark_scen1.txt
        other_data = [data for i, data in enumerate(micro_mAP_data) if i != 1]
    else:
        benchmark_data = None
        other_data = micro_mAP_data
    
    # Berechne Mittelwert, Minimum und Maximum der "anderen" Szenarien
    other_data_array = np.array(other_data)
    mean_values = np.mean(other_data_array, axis=0)
    min_values = np.min(other_data_array, axis=0)
    max_values = np.max(other_data_array, axis=0)
    
    # Spline-Glättung für Durchschnitt und Bereich (für den Plot)
    x_smooth, mean_smooth = smooth_curve(x_values, mean_values)
    _, min_smooth = smooth_curve(x_values, min_values)
    _, max_smooth = smooth_curve(x_values, max_values)
    
    # Ausgabe der unsmoothten Durchschnittswerte (ein Datenpunkt pro Runde)
    print("Average line data points (x, y):")
    for x_val, y_val in zip(x_values, mean_values):
        print(f"{x_val:.2f}, {y_val:.4f}")
    
    # Plot: Schattierter Bereich (min bis max) der anderen Szenarien
    plt.fill_between(x_smooth, min_smooth, max_smooth, color='lightblue', alpha=0.5,
                     label="Range (min-max)")
    
    # Plot: Durchschnittslinie der anderen Szenarien (Spline-geglättet)
    plt.plot(x_smooth, mean_smooth, color='blue', linewidth=2,
             label="Average micro mAP, LRP 97% prun + 5% retain, egge")
    
    # Plot: Benchmark-Daten (Spline-geglättet, k=3) als rote, gestrichelte Linie
    if benchmark_data is not None:
        x_bench, bench_smooth = smooth_curve(x_values, np.array(benchmark_data))
        plt.plot(x_bench, bench_smooth, linestyle='--', color='red', linewidth=2,
                 #label="Scenario 2 Benchmark (Spline k=3)")
                 label="Scenario 2 Benchmark")
    
    # Füge eine senkrechte graue, gestrichelte Linie bei Round 4 ein
    plt.axvline(x=12, color='gray', linestyle='-.', linewidth=1.5, label='Pruning Round')
    
    # Achsenbeschriftung und Grid
    plt.xlabel("Communication Round", fontsize=16)
    plt.ylabel("micro mAP", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Legend und Achsenticks formatieren
    plt.legend(fontsize=14, loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Layout optimieren und Plot speichern
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot micro mAP values from multiple log files.")
    parser.add_argument("files", nargs='+', type=str, help="Paths to the log files.")
    parser.add_argument("--labels", nargs='+', type=str, default=None, help="Labels for each scenario.")
    # Standard-Benchmark: fed_avg_benchmark_scen1.txt
    parser.add_argument("--scenario2_benchmark", type=str, default="fed_avg_benchmark_scen2.txt",
                        help="Path to the Scenario 2 Benchmark file (used here as Scenario 1 Benchmark).")
    args = parser.parse_args()
    
    # Extrahiere micro mAP Werte aus den Input-Dateien
    micro_mAP_data = []
    for file in args.files:
        micro_mAP_data.append(extract_micro_mAP(file))
    
    # Extrahiere Benchmark-Daten aus der Benchmark-Datei (fed_avg_benchmark_scen1.txt)
    scenario2benchmark = extract_micro_mAP(args.scenario2_benchmark)
    print(f"Scenario 2 Benchmark Data: {scenario2benchmark}")  # Debugging output
    if scenario2benchmark:
        micro_mAP_data.insert(1, scenario2benchmark)
    else:
        print("Warning: Scenario 2 Benchmark data is empty. Benchmark curve will not be plotted.")
    
    # Stelle sicher, dass alle Datensätze dieselbe Länge haben
    if micro_mAP_data:
        min_length = min(len(data) for data in micro_mAP_data)
        micro_mAP_data = [data[:min_length] for data in micro_mAP_data]
    else:
        print("No micro mAP data available. Exiting.")
        exit(1)
    
    # Labels
    if args.labels is None:
        labels = [f"Scenario {i+1}" for i in range(len(micro_mAP_data))]
    else:
        labels = args.labels
    
    # Falls Benchmark vorhanden, setze Label für den zweiten Datensatz
    if len(micro_mAP_data) > 1:
        labels[1] = "Scenario 1 Benchmark"
    
    print(f"Labels: {labels}")  # Debugging output
    
    # Plot und speichern
    output_plot_path = "micro_mAP_multi_8test_scenario2_with_benchmark.png"
    plot_micro_mAP(micro_mAP_data, labels=labels, output_path=output_plot_path)
    
    print(f"Plot saved to {output_plot_path}")
