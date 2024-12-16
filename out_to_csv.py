import re
import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Check if the file path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python script_name.py <path_to_out_file>")
    sys.exit(1)

# Get the input file path from the command-line argument
input_file_path = sys.argv[1]
output_csv_path = input_file_path.replace('.out', '_summary.csv')
output_plot_path = input_file_path.replace('.out', '_plot.png')

# Regex patterns to capture relevant information
round_pattern = re.compile(r"ROUND (\d+)/\d+")
micro_map_pattern = re.compile(r"micro\s+precision:.*?mAP:\s*([\d.]+)")
macro_map_pattern = re.compile(r"macro\s+precision:.*?mAP:\s*([\d.]+)")
pruning_mask_pattern = re.compile(r"Global Pruning Mask")
pruning_rate_pattern = re.compile(r"Pruning-rate:\s*([\d.]+)")

# Variables to store results
results = []
current_round = None
pruning_mask_round = None
pruning_rate = None

# Parse the .out file
try:
    with open(input_file_path, 'r') as file:
        for line in file:
            # Find and store the current round
            round_match = round_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                print(f"Processing ROUND {current_round}")

            # Find and store pruning mask round and rate
            pruning_mask_match = pruning_mask_pattern.search(line)
            pruning_rate_match = pruning_rate_pattern.search(line)
            if pruning_mask_match and current_round is not None:
                pruning_mask_round = current_round
            if pruning_rate_match:
                pruning_rate = float(pruning_rate_match.group(1))

            # Find micro and macro mAP values
            micro_map_match = micro_map_pattern.search(line)
            macro_map_match = macro_map_pattern.search(line)

            # Debugging: Check for matches
            if micro_map_match:
                print(f"Matched Micro mAP: {line.strip()}")
            if macro_map_match:
                print(f"Matched Macro mAP: {line.strip()}")

            # Store mAP values
            if micro_map_match or macro_map_match:
                micro_map = float(micro_map_match.group(1)) if micro_map_match else None
                macro_map = float(macro_map_match.group(1)) if macro_map_match else None
                results.append({
                    'Round': current_round,
                    'Micro mAP': micro_map,
                    'Macro mAP': macro_map,
                    'Pruning Mask Round': pruning_mask_round,
                    'Pruning Rate': pruning_rate
                })
                print(f"Appended to results: Round={current_round}, Micro mAP={micro_map}, Macro mAP={macro_map}")

except FileNotFoundError:
    print(f"Error: File '{input_file_path}' not found.")
    sys.exit(1)

# Check if results were collected
if not results:
    print("No results were collected. Check the parsing logic.")
    sys.exit(1)

# Write results to a CSV file
with open(output_csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=['Round', 'Micro mAP', 'Macro mAP', 'Pruning Mask Round', 'Pruning Rate'])
    csv_writer.writeheader()
    csv_writer.writerows(results)

# Load data into a DataFrame
df = pd.DataFrame(results, columns=['Round', 'Micro mAP', 'Macro mAP', 'Pruning Mask Round', 'Pruning Rate'])

# Debugging: Check DataFrame contents
if df.empty:
    print("Error: DataFrame is empty. Check the input file for proper formatting.")
    sys.exit(1)

# Plot mAP values
plt.figure(figsize=(10, 6))
plt.plot(df['Round'], df['Micro mAP'], label='Micro mAP', marker='o')
plt.plot(df['Round'], df['Macro mAP'], label='Macro mAP', marker='x')
plt.xlabel('Rounds')
plt.ylabel('mAP Values')
plt.title('mAP Values per Round')
plt.legend()
plt.grid(True)

# Save the plot as PNG
plt.savefig(output_plot_path)
print(f"Plot saved as: {output_plot_path}")

# Show the plot
plt.show()

# Inform the user about the generated CSV
print(f"Summary CSV file created: {output_csv_path}")
