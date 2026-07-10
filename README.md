# TAVI Calcium Topography Machine Learning Re-analysis

## Overview

This repository contains reproducible code for a secondary analysis of the Nuremberg TAVR cohort (Pollari et al., *J Cardiovasc Comput Tomogr*, 2020; Mendeley Data: doi:10.17632/dkr773wtn3.1).

### Study Objectives

Develop and internally validate interpretable machine learning models that integrate cusp-specific aortic valve and left ventricular outflow tract (LVOT) calcium topography with clinical, echocardiographic, and CT predictors for:

- **Primary endpoint:** Mid-term all-cause mortality after TAVI
- **Secondary endpoint:** VARC-2 composite of major periprocedural complications

## Design Principles (TRIPOD+AI / PROBAST+AI)

- **Unit of analysis:** Patient
- **Predictors:** Pre-procedural variables only (clinical, echocardiographic, MDCT, calcium topography, surgical risk scores, planned device/access)
- **Comparator:** EuroSCORE II
- **Missing data:** Imputed within each cross-validation fold to prevent data leakage
- **Validation:** Repeated stratified cross-validation with out-of-fold predictions

## Repository Structure

```text
.
├── common.py
├── 01_survival_mortality.py
├── 02_binary_complication.py
├── 03_tables.py
├── run_all.py
├── build_manuscript.py
├── build_supplementary.py
├── build_response.py
├── requirements.txt
├── rawdata.xlsx
├── references.md
├── .gitignore
└── LICENSE
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

Main packages:

- scikit-survival
- lifelines
- shap
- python-docx

### Add the dataset

Download `rawdata.xlsx` from the Mendeley Data repository and place it in the project root.

### Run the complete pipeline

```bash
python run_all.py
```

Outputs are written to:

- `figures/`
- `tables/`
- `results/`

### Generate the manuscript

```bash
python build_manuscript.py
```

This generates `Manuscript_TAVI_calcium_ML.docx`.

## Models

### Survival models

- EuroSCORE II (Cox PH)
- Penalized Cox (Coxnet)
- Gradient Boosted Survival
- Random Survival Forest
- Random Survival Forest (without calcium features)

Validation:
- 10 repeats × 5-fold stratified cross-validation (50 folds)
- C-index, time-dependent AUC, Integrated Brier Score, calibration, decision curve analysis, Kaplan-Meier analysis

### Binary complication models

- EuroSCORE II Logistic Regression
- L2 Logistic Regression
- Random Forest
- Histogram Gradient Boosting

Validation:
- 20 repeats × 5-fold stratified cross-validation
- AUROC, AUPRC, Brier score, calibration, diagnostic performance, decision curve analysis, SHAP, subgroup analyses

## Chunked Execution

```bash
python 01_survival_mortality.py chunk 0 4
python 01_survival_mortality.py chunk 5 9
python 01_survival_mortality.py aggregate
```

## Reproducibility

- Fixed random seed (`RANDOM_STATE = 20240607`)
- Deterministic cross-validation
- Saved out-of-fold predictions

## Citation

Please cite:

> Pollari F, Hitzl W, Vogt F, et al. Aortic valve calcification as a risk factor for major complications and reduced survival after transcatheter replacement. *J Cardiovasc Comput Tomogr.* 2020;14(4):307–313.

Methodological references:

- TRIPOD+AI (BMJ, 2024)
- PROBAST+AI (BMJ, 2025)
- Random Survival Forests (Ann Appl Stat, 2008)
- Decision Curve Analysis (Med Decis Making, 2006)

## License

MIT License. Original dataset (`rawdata.xlsx`) remains subject to the Mendeley Data CC BY 4.0 license.
