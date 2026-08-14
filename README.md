# SPE DSEATS Datathon 2026

This repository contains the code, data, and submissions for the SPE DSEATS Africa 2026 Datathon Competition.

## Overview
The goal of this project is to predict `Oil_Presence` using provided geological and physical datasets. The approach involves two types of submissions:
1. **Standard ML Submission (Blind)**: A purely data-driven approach using a Gradient Boosting Classifier.
2. **Physics-Informed Submission**: A domain-aware approach that applies physical rules (e.g., Shale permeability overrides, KNN imputation based on rock and trap types, and physical boundary clamping) before training a Support Vector Machine (SVM) model.

## Project Structure
- `jumah.py`: Python script containing the final pipeline for generating both the Standard and Physics-Informed submissions.
- `datathon_2026.ipynb`: Jupyter notebook used for exploratory data analysis (EDA) and model development.
- `oil_presence_trainset.csv`: Training dataset.
- `oil_presence_testset.csv`: Test dataset for evaluation.
- `SPE DSEATS Africa 2026 Datathon Competition Guidelines.pdf`: Official guidelines and rules for the datathon.

## Getting Started

### Prerequisites
Ensure you have the following Python libraries installed:
- `pandas`
- `scikit-learn`
- `jupyter` (if you want to run the notebook)

### Generating Submissions
To execute the pipeline and generate the final submission files (`submission_part1.csv` and `submission_part2.csv`), run the main script:
```bash
python jumah.py
```
