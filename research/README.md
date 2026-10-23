# MobilityMind Research Pipeline

## Purpose

This folder contains reproducible research utilities for studying:

1. Camera-based recognition of recorded fall events.
2. Movement characteristics associated with fall history.
3. Longer-term fall outcomes, if appropriate longitudinal data are available.

These are different research questions. Their results must not be treated
as interchangeable.

## Datasets

### UP-Fall

Purpose:
Evaluate camera-derived features for distinguishing recorded activities
and simulated fall events.

Limitations:
- Recorded falls are not the same as naturally occurring falls.
- Detection is not the same as predicting a future fall.
- Evaluation must keep participants separate between training and testing.
- Feature extraction and labels must be documented.

Official source:
https://sites.google.com/up.edu.mx/challenge-up-2019/data

### LTMM

Purpose:
Study associations between movement measurements, clinical assessments,
and reported fall history.

Limitations:
- The supplied faller/non-faller groups are based on past falls.
- This classification does not establish future-fall prediction.
- Raw accelerometer recordings need appropriate processing before use.
- Verify spreadsheet columns and label coding before analysis.

Official source:
https://physionet.org/content/ltmm/1.0.0/

### TILDA

Purpose:
Investigate associations between health/mobility measurements and
subsequent fall outcomes when suitable follow-up data are available.

Limitations:
- Access may require an application.
- Variables and coding must be verified against the data documentation.
- Only use predictor measurements collected before the outcome period.
- Do not use future-wave variables as predictors of an earlier outcome.

Official source:
https://tilda.tcd.ie/data/accessing-data/

## Installation

From the CongressionalApp project folder:

    python -m pip install -r research/requirements.txt

## Inventory local datasets

    python research/dataset_inventory.py

## Inspect LTMM clinical data

    python research/analyze_ltmm.py --file "research/data/ltmm/ClinicalDemogData_COFL.xlsx"

## Evaluation CSV format

The generic evaluator expects a CSV with:

- one row per sample or observation
- a participant ID column
- a binary target column
- numeric or categorical predictor columns

Do not use participant IDs as model features.

Do not include features that reveal the target label.

For UP-Fall, rows should represent documented video trials or carefully
defined time windows, with features extracted using a documented procedure.

For LTMM, rows should represent clearly defined participant-level
observations or windows, with labels verified from official documentation.

For TILDA, rows should define a prediction point and a later outcome period.
Predictor variables must come from before that outcome period.

## Example evaluation command

The following is only an example. It will work only after you have created
a real, correctly labeled CSV with the specified columns.

    python research/evaluate_grouped.py --csv research/data/example.csv --target fall_event --group participant_id --positive-label 1

## Evaluation

The evaluator saves:
- accuracy
- balanced accuracy
- precision
- sensitivity/recall
- specificity
- F1
- ROC-AUC
- false-positive and false-negative counts
- participant-level train/test split information
- predictions for the test set

The test set must not be used to choose features, tune thresholds, or
repeatedly improve the model.

For a stronger final evaluation, reserve an independent test group and
perform model development using training data only.

## Interpretation

UP-Fall results concern recorded fall-event recognition unless a different
outcome is explicitly defined.

LTMM results concern fall-history associations unless future outcomes are
available.

TILDA may support longitudinal outcome analysis when the correct follow-up
variables are available and temporally ordered.

Do not report a clinical risk score or claim that the system can predict
imminent falls without appropriate validation.

A model trained on these datasets is a research prototype, not a substitute
for professional assessment or emergency monitoring.
