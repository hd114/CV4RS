from openai import OpenAI
import os
import time
import subprocess
import glob
import argparse

# Setze den API-Schlüssel aus der Umgebungsvariable
skey = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=skey)

MODEL = "gpt-4o"  # Modell, das verwendet werden soll

# Pfade zu den relevanten Dateien und Ordnern
project_dir = "."  # Pfad zu deinem Projekt
train_script = os.path.join(project_dir, "train_script.slurm")
log_dir = os.path.join(project_dir, "log")
code_file = os.path.join(project_dir, "train.py")  # Haupt-Python-Datei


def submit_job():
    """Startet das Training mit sbatch und gibt die Job-ID zurück."""
    try:
        result = subprocess.run(["sbatch", train_script], capture_output=True, text=True)
        output = result.stdout.strip()
        job_id = output.split()[-1]  # Extrahiere die Job-ID
        return job_id
    except Exception as e:
        print(f"Fehler beim Starten des Jobs: {e}")
        return None


def wait_for_job(job_id):
    """Wartet, bis der Slurm-Job abgeschlossen ist."""
    while True:
        try:
            result = subprocess.run(["squeue", "--job", job_id], capture_output=True, text=True)
            if job_id not in result.stdout:
                break  # Job ist abgeschlossen
            time.sleep(10)  # 10 Sekunden warten
        except Exception as e:
            print(f"Fehler beim Überprüfen des Jobstatus: {e}")
            break


def get_relevant_logs(log, num_lines):
    """Extrahiert die letzten Zeilen eines Logs."""
    lines = log.splitlines()
    if len(lines) <= num_lines:
        return log  # Log ist bereits kurz genug
    else:
        return "\n".join(lines[-num_lines:])  # Nur die letzten num_lines Zeilen


def get_latest_logs(log_dir):
    """Findet die neuesten .err- und .out-Dateien im log-Verzeichnis und extrahiert relevante Zeilen."""
    try:
        err_files = sorted(glob.glob(os.path.join(log_dir, "pytorch_job_*.err")), key=os.path.getmtime)
        out_files = sorted(glob.glob(os.path.join(log_dir, "pytorch_job_*.out")), key=os.path.getmtime)

        if not err_files or not out_files:
            print("Keine Log-Dateien gefunden.")
            return "", ""

        latest_err = err_files[-1]
        latest_out = out_files[-1]

        with open(latest_err, "r") as err_file:
            err_content = err_file.read()
        with open(latest_out, "r") as out_file:
            out_content = out_file.read()

        # Nur die letzten relevanten Zeilen extrahieren
        err_relevant = get_relevant_logs(err_content, num_lines=30)
        out_relevant = get_relevant_logs(out_content, num_lines=50)

        return err_relevant, out_relevant
    except Exception as e:
        print(f"Fehler beim Lesen der Logs: {e}")
        return "", ""


def improve_code(current_code, logs):
    """Verbessert den Code basierend auf Logs."""
    err_logs, out_logs = logs

    prompt = f"""
    Hier ist der aktuelle Python-Code, der verbessert werden soll:
    ```python
    {current_code}
    ```

    Hier sind die letzten relevanten Fehlerlogs (stderr):
    ```plaintext
    {err_logs}
    ```

    Hier sind die letzten relevanten Ausgabelogs (stdout):
    ```plaintext
    {out_logs}
    ```

    Deine Aufgabe:
    - Identifiziere die Probleme basierend auf den Logs.
    - Verbessere den Code, um die Fehler zu beheben und ihn effizienter zu machen.
    - Füge präzise Kommentare hinzu, um jede Änderung zu erklären.
    - Nimm nur die notwendigen Änderungen vor.
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein Assistent, der beim Debugging und Optimieren von Python-Code hilft."},
                {"role": "user", "content": prompt},
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fehler bei der Code-Optimierung: {e}")
        return current_code


def read_code(file_path):
    """Liest den aktuellen Code aus der Datei."""
    with open(file_path, "r") as file:
        return file.read()


def write_code(file_path, new_code):
    """Schreibt den aktualisierten Code in die Datei."""
    with open(file_path, "w") as file:
        file.write(new_code)


def main(max_iterations):
    """Hauptfunktion für den Feedback-Loop."""
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # Job starten
        job_id = submit_job()
        if not job_id:
            print("Job konnte nicht gestartet werden. Abbruch.")
            break

        print(f"Job {job_id} gestartet. Warte auf Abschluss...")
        wait_for_job(job_id)

        # Logs lesen
        logs = get_latest_logs(log_dir)
        if not logs[0] and not logs[1]:
            print("Keine Logs gefunden. Abbruch.")
            break

        # Aktuellen Code lesen
        current_code = read_code(code_file)

        # Neuen Code generieren
        new_code = improve_code(current_code, logs)

        # Änderungen anwenden
        write_code(code_file, new_code)

        print(f"Iteration {iteration + 1} abgeschlossen. Code aktualisiert.\n")

    print("Loop beendet. Prüfe die Logs für Details.")


if __name__ == "__main__":
    # Argumente aus dem Terminal parsen
    parser = argparse.ArgumentParser(description="Feedback-Loop für Slurm-basierte Jobs.")
    parser.add_argument(
        "max_iterations",
        type=int,
        help="Die maximale Anzahl der Iterationen für den Feedback-Loop."
    )
    args = parser.parse_args()

    # Hauptfunktion ausführen
    main(args.max_iterations)
