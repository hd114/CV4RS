import argparse

def find_max_micro_map(file_path, max_round):
    """
    Find the largest micro mAP value and its corresponding line number in a file,
    up to a specified communication round.

    Args:
        file_path (str): Path to the .out file.
        max_round (int): Maximum communication round to consider.

    Returns:
        tuple: (max_map_value, max_map_line_number, max_map_round)
    """
    max_map_value = 0
    max_map_line_number = None
    max_map_round = None  # Speichert die Runde mit dem höchsten Wert
    current_round = None  # Speichert die aktuelle Runde

    try:
        with open(file_path, "r") as file:
            for line_number, line in enumerate(file, start=1):
                
                # Prüfen, ob eine neue Runde beginnt (Format: "Round X/Y")
                if "Round" in line and "/" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() == "round" and i + 1 < len(parts):
                                round_info = parts[i + 1].split("/")  # Extrahiert X aus "X/Y"
                                if round_info[0].isdigit():
                                    current_round = int(round_info[0])  # Speichert aktuelle Runde
                                    break
                    except (IndexError, ValueError):
                        continue  # Falls Format nicht passt, ignoriere Zeile
                
                # Prüfen, ob es eine micro mAP Zeile ist
                if "micro" in line and "mAP:" in line:
                    try:
                        # Falls die aktuelle Runde existiert und größer als max_round ist -> ignorieren
                        if current_round is not None and current_round > max_round:
                            continue
                        
                        # Extrahiere den mAP-Wert
                        parts = line.split("mAP:")
                        map_value = float(parts[1].strip().split()[0])

                        if map_value > max_map_value:
                            max_map_value = map_value
                            max_map_line_number = line_number
                            max_map_round = current_round  # Speichere die Runde mit dem höchsten Wert
                    except (IndexError, ValueError):
                        continue  # Falls Format nicht passt, ignoriere Zeile

        print(f"Max micro mAP: {max_map_value:.4f} at line {max_map_line_number}, Round {max_map_round} (up to round {max_round})")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# CLI Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the highest micro mAP value and its line number in a .out file up to a specified round.")
    parser.add_argument("file", type=str, help="Path to the .out file")
    parser.add_argument("--max_round", type=int, required=True, help="Maximum communication round to consider")

    args = parser.parse_args()
    find_max_micro_map(args.file, args.max_round)
