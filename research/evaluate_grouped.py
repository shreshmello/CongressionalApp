from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--positive-label", default="1")
    parser.add_argument(
        "--output",
        default="research/outputs/evaluation.json"
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)

    for column in [args.target, args.group]:
        if column not in data.columns:
            raise SystemExit(f"Required column missing: {column}")

    data = data.dropna(subset=[args.target, args.group]).copy()

    labels = data[args.target].astype(str)
    classes = sorted(labels.unique())

    if len(classes) != 2:
        raise SystemExit(
            f"Expected two classes, but found: {classes}"
        )

    positive_label = str(args.positive_label)

    if positive_label not in classes:
        raise SystemExit(
            f"Positive label {positive_label} not found in {classes}"
        )

    y = (labels == positive_label).astype(int).to_numpy()
    groups = data[args.group].astype(str).to_numpy()

    excluded = {
        args.target,
        args.group,
        "video_path",
        "record_path",
    }

    feature_data = data[
        [column for column in data.columns if column not in excluded]
    ].copy()

    feature_data = feature_data.dropna(axis=1, how="all")

    feature_data = feature_data.loc[
        :,
        [
            column
            for column in feature_data.columns
            if feature_data[column].nunique(dropna=True) > 1
        ],
    ]

    if feature_data.empty:
        raise SystemExit("No usable feature columns were found.")

    categorical = feature_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric = [
        column for column in feature_data.columns
        if column not in categorical
    ]

    transformers = []

    if numeric:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric)
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            )
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

    if len(np.unique(groups)) < 4:
        raise SystemExit(
            "At least four unique participant groups are required "
            "for this baseline grouped holdout."
        )

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=args.test_size,
        random_state=args.seed,
    )

    split_found = None

    for train_indices, test_indices in splitter.split(
        feature_data, y, groups
    ):
        if (
            len(np.unique(y[train_indices])) == 2
            and len(np.unique(y[test_indices])) == 2
        ):
            split_found = (train_indices, test_indices)
            break

    if split_found is None:
        raise SystemExit(
            "Could not find a participant-grouped split containing "
            "both classes in training and testing. Check your labels "
            "and how they are distributed across participants."
        )

    train_indices, test_indices = split_found

    X_train = feature_data.iloc[train_indices]
    X_test = feature_data.iloc[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if tn + fp else None

    try:
        auc = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        auc = None

    results = {
        "task": "Binary classification of the supplied labels",
        "target": args.target,
        "positive_label": positive_label,
        "group_column": args.group,
        "total_rows": len(data),
        "total_participants": len(np.unique(groups)),
        "training_rows": len(train_indices),
        "testing_rows": len(test_indices),
        "training_participants": sorted(
            np.unique(groups[train_indices]).tolist()
        ),
        "testing_participants": sorted(
            np.unique(groups[test_indices]).tolist()
        ),
        "features": feature_data.columns.tolist(),
        "metrics": {
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
            "specificity": float(specificity) if specificity is not None else None,
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "warning": (
            "This is a baseline grouped holdout, not clinical validation. "
            "Historical fall status does not mean future-fall prediction."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to {output.resolve()}")


if __name__ == "__main__":
    main()
