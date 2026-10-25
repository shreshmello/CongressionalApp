from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

files = []

if DATA.exists():
    for path in sorted(DATA.rglob("*")):
        if path.is_file():
            files.append({
                "relative_path": str(path.relative_to(DATA)),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size
            })

csv_path = OUTPUTS / "dataset_inventory.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["relative_path", "extension", "size_bytes"]
    )
    writer.writeheader()
    writer.writerows(files)

json_path = OUTPUTS / "dataset_inventory.json"
json_path.write_text(json.dumps(files, indent=2), encoding="utf-8")

print(f"Found {len(files)} files.")
print(f"Inventory saved to: {OUTPUTS}")

if not files:
    print("No datasets found. Download data into research/data/")
