import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def read_data(filename):
    """
    Reads the data points from the given file.
    Expects the file to contain a header line and then lines of the form:
        1.00, 0.5197
        2.00, 0.6086
        ...
    
    Args:
        filename (str): Path to the file.
    
    Returns:
        tuple: Two numpy arrays (x_values, y_values).
    """
    x_values = []
    y_values = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip leere Zeilen oder Header-Zeilen
            if not line or line.startswith("Average line data points"):
                continue
            # Zerlege die Zeile an dem Komma
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                x_values.append(x)
                y_values.append(y)
            except ValueError:
                # Falls die Konvertierung fehlschlägt, überspringe die Zeile
                continue
    return np.array(x_values), np.array(y_values)

def plot_cubic_spline_from_file(filename):
    """
    Reads the data from the given file, interpolates using cubic spline (k=3)
    und plottet die Originaldaten sowie die interpolierte Kurve.
    
    Args:
        filename (str): Path to the data file.
    """
    # Daten einlesen
    x, y = read_data(filename)
    
    # Erstelle den Cubic Spline (k=3)
    spline = make_interp_spline(x, y, k=3)
    
    # Generiere interpolierte x-Werte (z.B. 300 Punkte für einen glatten Plot)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = spline(x_smooth)
    
    # Erstelle den Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_smooth, y_smooth, label="Cubic Spline Interpolation (k=3)", color="blue")
    plt.scatter(x, y, color="red", label="Original Data Points")
    plt.xlabel("Communication Round", fontsize=14)
    plt.ylabel("Average micro mAP", fontsize=14)
    plt.title("Cubic Spline Interpolation from fed_avg_benchmark.txt", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("fed_avg_benchmark_spline.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_cubic_spline_from_file("fed_avg_benchmark.txt")
