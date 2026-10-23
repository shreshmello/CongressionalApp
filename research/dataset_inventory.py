from pathlib import Path
import json
import csv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = BASE_DIR / "dataset_inventory.json"

DATASETS = {
    "UP-Fall": DATA_DIR / "upfall",
    "LTMM": DATA_DIR / "ltmm",
    "TILDA": DATA_DIR / "tilda",
}


def inspect_dataset(name, folder):
    if not folder.exists():
        return {
            "dataset": name,
            "folder_exists": False,
            "file_count": 0,
            "files": [],
        }

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
    )

    return {
        "dataset": name,
        "folder_exists": True,
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "files": [
            {
                "name": str(p.relative_to(DATA_DIR)),
                "bytes": p.stat().st_size,
                "extension": p.suffix.lower(),
            }
            for p in files[:500]
        ],
        "file_listing_truncated": len(files) > 500,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "purpose": "Local research-data inventory",
        "generated_files_are_not_research_results": True,
        "datasets": [
            inspect_dataset(name, folder)
            for name, folder in DATASETS.items()
        ],
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    for item in report["datasets"]:
        print(
            f"{item['dataset']}: "
            f"{item['file_count']} files found"
        )

    print(f"\nInventory saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
