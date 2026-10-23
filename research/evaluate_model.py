"""
Research-only evaluation for a labeled fall-event dataset.

Input CSV requirements:
- One row per independent observation/window.
- subject_id: participant identifier.
- label: 1 for a labeled fall event, 0 for non-fall activity.
- One or more numeric feature columns.
- Do not use random row splitting when rows from the same participant
  can appear in both sets.

IMPORTANT:
This script evaluates a dataset you supply. It does not train on the live
browser features automatically and does not establish clinical validity.

A fall-event dataset is not the same as a dataset of pre-fall instability
or future fall risk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


def specificity_score(y_true, y_pred):
    """Calculate specificity, or the true-negative rate."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    if (tn + fp) == 0:
        return None

    return float(tn / (tn + fp))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a research fall-event classifier "
            "with participant-level holdout."
        )
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with subject_id, label, and numeric feature columns.",
    )

    parser.add_argument(
        "--out",
        default="evaluation_output",
        help="Output directory.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of participants held out for testing.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the binary label.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required = {"subject_id", "label"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["subject_id", "label"]
    ).copy()

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    df = df[df["label"].isin([0, 1])].copy()

    if df["label"].nunique() != 2:
        raise SystemExit(
            "The dataset must contain both label 0 "
            "(non-fall) and label 1 (fall event)."
        )

    feature_cols = [
        column
        for column in df.columns
        if column not in {"subject_id", "label"}
    ]

    if not feature_cols:
        raise SystemExit(
            "No feature columns found. "
            "Add numeric movement-feature columns."
        )

    for column in feature_cols:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove features that contain no usable numeric values.
    feature_cols = [
        column
        for column in feature_cols
        if df[column].notna().any()
    ]

    if not feature_cols:
        raise SystemExit(
            "No usable numeric feature columns found."
        )

    X = df[feature_cols]
    y = df["label"].astype(int)
    groups = df["subject_id"].astype(str)

    if groups.nunique() < 4:
        raise SystemExit(
            "At least 4 distinct participants are recommended "
            "for a simple grouped holdout; more is better."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit(
            "The participant-level split left one partition "
            "with only one class. Try a different dataset "
            "or test-size."
        )

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    classifier = model.named_steps["classifier"]

    class_one_index = list(
        classifier.classes_
    ).index(1)

    probabilities = model.predict_proba(
        X_test
    )[:, class_one_index]

    predictions = (
        probabilities >= args.threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    report = {
        "purpose": (
            "Research-only fall-event classification evaluation; "
            "not a validated clinical tool."
        ),

        "target": (
            "Labeled fall event in the supplied row/window, "
            "NOT imminent fall prediction or future fall risk."
        ),

        "split": (
            "GroupShuffleSplit by subject_id to reduce "
            "participant leakage."
        ),

        "participants_total": int(groups.nunique()),

        "participants_train": int(
            groups.iloc[train_idx].nunique()
        ),

        "participants_test": int(
            groups.iloc[test_idx].nunique()
        ),

        "rows_train": int(len(train_idx)),
        "rows_test": int(len(test_idx)),

        "features": feature_cols,
        "threshold": args.threshold,

        "metrics": {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),

            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_test,
                    predictions,
                )
            ),

            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

            "recall_sensitivity": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

            "specificity": specificity_score(
                y_test,
                predictions,
            ),

            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

            "false_positives": int(fp),
            "false_negatives": int(fn),

            "true_positives": int(tp),
            "true_negatives": int(tn),

            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),
        },

        "limitations": [
            (
                "Performance depends on the dataset, labels, "
                "camera/sensor setup, and participants."
            ),
            (
                "Simulated falls and young healthy participants "
                "may not represent real-world older-adult falls."
            ),
            (
                "Event detection is not the same as predicting "
                "a fall before it happens."
            ),
            (
                "Do not display the model output as a medical risk "
                "score without external validation and clinical review."
            ),
        ],
    }

    metrics_path = out_dir / "metrics.json"

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    predictions_path = out_dir / "test_predictions.csv"

    pd.DataFrame(
        {
            "subject_id": groups.iloc[test_idx].to_numpy(),
            "label": y_test.to_numpy(),
            "predicted_label": predictions,
            "predicted_probability": probabilities,
        }
    ).to_csv(
        predictions_path,
        index=False,
    )

    model_path = out_dir / "research_model.joblib"

    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": feature_cols,
            "threshold": args.threshold,
        },
        model_path,
    )

    print(json.dumps(report, indent=2))

    print(
        f"\nSaved evaluation files to: {out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
