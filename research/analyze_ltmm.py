import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LTMM clinical spreadsheet columns."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to ClinicalDemogData_COFL.xlsx",
    )

    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or zero-based sheet index.",
    )

    parser.add_argument(
        "--output",
        default="research/results/ltmm",
    )

    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find spreadsheet: {file_path}"
        )

    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    df = pd.read_excel(file_path, sheet_name=sheet)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the column inventory.
    columns = pd.DataFrame({
        "column": df.columns.astype(str),
        "data_type": [
            str(df[c].dtype) for c in df.columns
        ],
        "non_missing_count": [
            int(df[c].notna().sum()) for c in df.columns
        ],
        "unique_values": [
            int(df[c].nunique(dropna=True)) for c in df.columns
        ],
    })

    columns.to_csv(
        output_dir / "ltmm_columns.csv",
        index=False,
    )

    # Summarize numeric variables without assuming that
    # any particular column is a valid fall-risk outcome.
    numeric = df.select_dtypes(include="number")

    if not numeric.empty:
        numeric.describe().T.to_csv(
            output_dir / "ltmm_numeric_summary.csv"
        )

    print("\nSpreadsheet shape:", df.shape)
    print("\nAvailable columns:\n")

    for column in df.columns:
        print(
            f"- {column} "
            f"(type={df[column].dtype}, "
            f"non-missing={df[column].notna().sum()})"
        )

    print("\nSaved column inventory to:")
    print(output_dir / "ltmm_columns.csv")

    print("\nNext step:")
    print(
        "Review the official data documentation, identify the "
        "participant ID and fall-history columns, and verify their "
        "coding before preparing a labeled research CSV."
    )


if __name__ == "__main__":
    main()
