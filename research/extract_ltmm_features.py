from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import wfdb


def summarize_signal(signal, prefix):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        return {}

    return {
        f"{prefix}_mean": float(np.mean(signal)),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_median_abs": float(np.median(np.abs(signal))),
        f"{prefix}_rms": float(np.sqrt(np.mean(signal ** 2))),
        f"{prefix}_range": float(np.ptp(signal)),
        f"{prefix}_diff_std": (
            float(np.std(np.diff(signal)))
            if len(signal) > 1 else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300,
    )

    args = parser.parse_args()

    root = Path(args.data_root)
    headers = sorted(root.rglob("*.hea"))

    if not headers:
        raise SystemExit(
            f"No WFDB .hea files found under {root}. "
            "Check that the dataset is downloaded and extracted."
        )

    rows = []

    for header in headers:
        try:
            record = wfdb.rdrecord(
                str(header.with_suffix("")),
                physical=True,
            )

            signal = np.asarray(record.p_signal)
            frequency = float(record.fs or 1.0)

            if signal.ndim == 1:
                signal = signal[:, None]

            max_samples = int(args.max_seconds * frequency)
            signal = signal[:max_samples]

            row = {
                "record_id": header.stem,
                "sampling_frequency_hz": frequency,
                "duration_seconds": len(signal) / frequency,
                "channel_count": signal.shape[1],
            }

            for index in range(signal.shape[1]):
                row.update(
                    summarize_signal(
                        signal[:, index],
                        f"channel_{index + 1}",
                    )
                )

            if signal.shape[1] >= 3:
                magnitude = np.sqrt(
                    np.sum(signal[:, :3] ** 2, axis=1)
                )

                row.update(
                    summarize_signal(magnitude, "magnitude")
                )

            rows.append(row)

            print(f"Processed {header.name}")

        except Exception as error:
            print(f"Skipped {header}: {error}")

    if not rows:
        raise SystemExit("No records were processed.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(output, index=False)

    print(f"Saved features to: {output.resolve()}")
    print(
        "Note: these features do not predict future falls. "
        "A documented participant label is still needed for evaluation."
    )


if __name__ == "__main__":
    main()
