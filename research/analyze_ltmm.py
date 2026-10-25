from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)

    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        raise SystemExit(f"Spreadsheet not found: {file_path}")

    workbook = pd.ExcelFile(file_path)

    print("Sheets:", workbook.sheet_names)

    for sheet in workbook.sheet_names:
        data = pd.read_excel(file_path, sheet_name=sheet)

        print(f"\n--- {sheet} ---")
        print("Rows:", len(data))
        print("Columns:", list(data.columns))

        print("\nFirst five rows:")
        print(data.head().to_string(index=False))

        print("\nMissing values:")
        print(data.isna().sum().to_string())


if __name__ == "__main__":
    main()
