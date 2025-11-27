import csv
import json
import os
from pathlib import Path

# ---------------------------
# OUTPUT
# ---------------------------
OUTPUT_FILE = "../data/processed/logs.jsonl"

def load_csv_file(path):
    """Generator: Reads a CSV file and yields each row as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def write(output):
    """Append a JSON object to the output JSONL file."""
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")

# ---------------------------
# LOGS → Anomaly Detection
# ---------------------------
def process_logs():
    root = "../data/raw/logs/"
    if not os.path.exists(root):
        print("Logs folder does not exist:", root)
        return

    for file in os.listdir(root):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(root, file)
        print("Processing logs:", file_path)

        for row in load_csv_file(file_path):
            # Build a single log string
            log_text = (
                f"[{row['timestamp']}] "
                f"{row['log_level']} "
                f"[{row['pod_name']}] "
                f"{row['message']}"
            )

            # Simple placeholder label for synthetic logs
            anomaly_label = "Anomaly detected." if row["log_level"].upper() == "ERROR" else "No anomaly detected."

            out = {
                "instruction": "Analyze the following Kubernetes logs and detect anomalies.",
                "input": log_text,
                "output": anomaly_label
            }

            write(out)

# ---------------------------
# POLICIES → Suggestions
# ---------------------------
def process_policies():
    root = "../data/raw/policies/"
    if not os.path.exists(root): return

    for file in os.listdir(root):
        if not file.endswith(".csv"): continue

        file_path = os.path.join(root, file)
        print("Processing policies:", file_path)

        for row in load_csv_file(file_path):
            out = {
                "instruction": "Analyze this enterprise policy document and suggest improvements.",
                "input": row.get("content", ""),
                "output": row.get("improvement", "")
            }
            write(out)

# ---------------------------
# MANIFESTS → Config Risks
# ---------------------------
def process_manifests():
    root = "../data/raw/manifests/"
    if not os.path.exists(root): return

    for file in os.listdir(root):
        if not file.endswith(".csv"): continue

        file_path = os.path.join(root, file)
        print("Processing manifests:", file_path)

        for row in load_csv_file(file_path):
            out = {
                "instruction": "Identify risks or misconfigurations in the Kubernetes manifest.",
                "input": row.get("yaml", ""),
                "output": row.get("analysis", "")
            }
            write(out)

# ---------------------------
# QA Reasoning → General SFT
# ---------------------------
def process_qa():
    root = "../data/raw/qa/"
    if not os.path.exists(root): return

    for file in os.listdir(root):
        if not file.endswith(".csv"): continue

        file_path = os.path.join(root, file)
        print("Processing QA:", file_path)

        for row in load_csv_file(file_path):
            out = {
                "instruction": row.get("question", ""),
                "input": "",
                "output": row.get("answer", "")
            }
            write(out)

# ---------------------------
# MAIN
# ---------------------------
def main():
    Path("../data/processed/").mkdir(parents=True, exist_ok=True)
    open(OUTPUT_FILE, "w").close()  # empty output file

    print("Processing logs...")
    process_logs()

    # print("Processing policies...")
    # process_policies()

    # print("Processing manifests...")
    # process_manifests()

    # print("Processing QA...")
    # process_qa()

    print("DONE → Written:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
