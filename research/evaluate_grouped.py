import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a classifier using participant-independent testing."
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--positive-label", required=True)
    parser.add_argument("--output", default="research/results")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional columns to exclude from model features.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.1 <= args.test_size <= 0.5:
        raise ValueError("--test-size must be between 0.1 and 0.5.")

    data_path = Path(args.csv).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    required = {args.target, args.group}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.dropna(subset=[args.target, args.group]).copy()

    if df.empty:
        raise ValueError("No rows remain after removing missing labels/groups.")

    # This script supports binary classification only.
    labels = list(pd.unique(df[args.target]))

    if len(labels) != 2:
        raise ValueError(
            f"Expected exactly 2 target classes, found {labels}. "
            "Check your labels and target definition."
        )

    if args.positive_label not in [str(v) for v in labels]:
        raise ValueError(
            f"Positive label {args.positive_label!r} not found in "
            f"target values {labels}."
        )

    # Explicitly convert the target to a binary outcome.
    y = (
        df[args.target].astype(str) == args.positive_label
    ).astype(int)

    if y.nunique() != 2:
        raise ValueError("Both positive and negative classes are required.")

    excluded = {
        args.target,
        args.group,
        *args.exclude,
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    if not feature_cols:
        raise ValueError("No feature columns remain.")

    X = df[feature_cols].copy()

    # Drop columns that are entirely missing.
    X = X.dropna(axis=1, how="all")
    feature_cols = list(X.columns)

    if X.empty or not feature_cols:
        raise ValueError("No usable features were found.")

    # Reject columns that could leak the answer directly.
    # Inspect all remaining features manually as well.
    print("Features used:", feature_cols)
    print("Target:", args.target)
    print("Group column:", args.group)

    # Participant-level split: a person must not appear in both sets.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.seed,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=df[args.group])
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(df[args.group].iloc[train_idx])
    test_groups = set(df[args.group].iloc[test_idx])

    if train_groups & test_groups:
        raise RuntimeError("Participant leakage detected!")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "The participant split does not contain both classes in "
            "training and testing. A valid evaluation cannot be calculated "
            "from this split. Review class distribution and participants."
        )

    numeric_cols = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        c for c in feature_cols
        if c not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessing = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=args.seed,
                min_samples_leaf=2,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) else None
    )

    metrics = {
        "experiment": data_path.stem,
        "dataset_file": str(data_path),
        "target_column": args.target,
        "group_column": args.group,
        "positive_label": args.positive_label,
        "features": feature_cols,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_idx)),
        "rows_test": int(len(test_idx)),
        "unique_train_participants": int(len(train_groups)),
        "unique_test_participants": int(len(test_groups)),
        "train_test_participant_overlap": len(
            train_groups & test_groups
        ),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "sensitivity_recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "specificity": (
            float(specificity) if specificity is not None else None
        ),
        "f1": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_class_counts": {
            "negative": int((y_test == 0).sum()),
            "positive": int((y_test == 1).sum()),
        },
        "interpretation_warning": (
            "Metrics describe this test split only. They do not establish "
            "clinical validity or future-fall prediction."
        ),
    }

    predictions_out = df.iloc[test_idx][
        [args.group, args.target]
    ].copy()

    predictions_out["actual_binary"] = y_test.to_numpy()
    predictions_out["predicted_binary"] = predictions
    predictions_out["predicted_probability"] = probabilities

    predictions_out.to_csv(
        output_dir / f"{data_path.stem}_test_predictions.csv",
        index=False,
    )

    with open(
        output_dir / f"{data_path.stem}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print("\nEVALUATION RESULTS")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
