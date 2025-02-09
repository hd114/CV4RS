import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Manuell extrahierte (x, y)-Werte für die zwei blauen Kurven
x_values = np.linspace(0, 1, 24)

# Erste blaue Kurve (durchgezogen)
y1_values = np.array([
    1.00, 0.98, 0.95, 0.92, 0.88, 0.84, 0.78, 0.72, 0.65, 0.58, 
    0.51, 0.45, 0.40, 0.36, 0.33, 0.30, 0.28, 0.27, 0.26, 0.25,
    0.24, 0.24, 0.24, 0.24
])

# Zweite blaue Kurve (gestrichelt)
y2_values = np.array([
    1.00, 0.95, 0.90, 0.85, 0.78, 0.70, 0.60, 0.50, 0.42, 0.35,
    0.29, 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13,
    0.12, 0.12, 0.12, 0.12
])

# Spline-Interpolation für glattere Kurven
x_smooth = np.linspace(0, 1, 300)
spline1 = make_interp_spline(x_values, y1_values, k=3)
spline2 = make_interp_spline(x_values, y2_values, k=3)

y1_smooth = spline1(x_smooth)
y2_smooth = spline2(x_smooth)

# Erstelle den Plot
plt.figure(figsize=(7, 4))

# Zeichne die zwei blauen Kurven mit exakten Werten und Spline-Glättung
plt.plot(x_smooth, y1_smooth, color="blue", linewidth=2, label="Solid Blue Line")
plt.plot(x_smooth, y2_smooth, linestyle="--", color="blue", linewidth=2, label="Dashed Blue Line")

# Horizontale graue Linien zur Orientierung
plt.axhline(y=1.0, color="gray", linestyle="-", linewidth=1.5, alpha=0.7)
plt.axhline(y=0.75, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
plt.axhline(y=0.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

# Diagramm-Stil
plt.title("Convolution Filters of ResNet-18", fontsize=14)
plt.xlabel("Normalized Pruning Rate")
plt.ylabel("Filter Response")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(False)

# Speichern als PNG-Datei
output_path = "convolution_filters_resnet18_blue.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

# Ausgabe des Speicherorts
output_path
