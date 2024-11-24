import yaml
import subprocess

# Lade die yml-Datei
with open("environment.yml", "r") as file:
    yml_data = yaml.safe_load(file)

# Extrahiere die Pakete
required_packages = yml_data["dependencies"]

# Füge pip-Pakete hinzu, falls vorhanden
pip_packages = []
for dep in required_packages:
    if isinstance(dep, dict) and "pip" in dep:
        pip_packages.extend(dep["pip"])
    elif isinstance(dep, str):
        pip_packages.append(dep)

# Liste installierte Pakete in der Umgebung auf
installed_packages = subprocess.check_output(["micromamba", "list"], text=True)

# Überprüfe jedes Paket
missing_packages = []
for pkg in pip_packages:
    if pkg.split("==")[0] not in installed_packages:
        missing_packages.append(pkg)

if missing_packages:
    print("Die folgenden Pakete fehlen in deiner Umgebung:")
    print("\n".join(missing_packages))
else:
    print("Alle Pakete aus der yml-Datei sind installiert.")
