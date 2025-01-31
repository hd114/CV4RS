import argparse

def find_max_micro_map(file_path):
    """
    Find the largest micro mAP value and its corresponding line number in a file.

    Args:
        file_path (str): Path to the .out file.

    Returns:
        tuple: (max_map_value, max_map_line_number)
    """
    max_map_value = 0
    max_map_line_number = None

    try:
        with open(file_path, "r") as file:
            for line_number, line in enumerate(file, start=1):
                if "micro" in line and "mAP:" in line:
                    try:
                        parts = line.split("mAP:")
                        map_value = float(parts[1].strip().split()[0])
                        if map_value > max_map_value:
                            max_map_value = map_value
                            max_map_line_number = line_number
                    except (IndexError, ValueError):
                        continue  # Ignore lines that don't match the expected format

        print(f"Max micro mAP: {max_map_value:.4f} at line {max_map_line_number}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# CLI Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the highest micro mAP value and its line number in a .out file.")
    parser.add_argument("file", type=str, help="Path to the .out file")

    args = parser.parse_args()
    find_max_micro_map(args.file)
