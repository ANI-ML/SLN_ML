# CT Radiomics for Canine LN Metastasis Prediction

This repository contains the code, data-processing pipeline, and analysis for
the manuscript *"CT-based radiomic features predict cervical lymph node
metastasis in dogs with oral malignancy: a machine-learning study using
leave-one-patient-out cross-validation."*

## Overview

The project analyses CT images from 49 dogs with oral tumors, segmenting
bilateral mandibular and retropharyngeal lymph nodes (195 observations,
18 metastatic, 9.2%). PyRadiomics extracts 107 features per node.
Feature selection (variance + Spearman-|r|>0.95 + Mann–Whitney U with
Benjamini–Hochberg FDR) and model training are performed **inside each
leave-one-patient-out cross-validation fold on training data only**, so the
held-out patient never contributes to selection, scaling, or SMOTE
resampling. Four classifiers are compared (Logistic Regression, Random
Forest, SVM, XGBoost).

Across the 49 nested-LOOCV folds, the four most consistently selected
features are texture-based and reproduce the pattern reported at the
full-data level:

| Feature | Folds selected |
|---|---|
| GLSZM LowGrayLevelZoneEmphasis | 49 / 49 |
| GLSZM SmallAreaLowGrayLevelEmphasis | 49 / 49 |
| GLCM MCC | 48 / 49 |
| GLSZM HighGrayLevelZoneEmphasis | 45 / 49 |

## Key Results (nested LOOCV, patient-level bootstrap 95% CIs)

| Model | AUC | 95% CI | Sensitivity | Specificity | PPV | NPV | Youden thr |
|---|---|---|---|---|---|---|---|
| Random Forest | **0.649** | 0.474–0.831 | 0.444 | 0.898 | 0.308 | 0.941 | 0.481 |
| XGBoost | 0.631 | 0.501–0.772 | 0.944 | 0.379 | 0.134 | **0.985** | 0.063 |
| Logistic Regression | 0.522 | 0.322–0.746 | 0.833 | 0.305 | 0.109 | 0.947 | 0.289 |
| SVM | 0.443 | 0.265–0.678 | 0.222 | 0.819 | 0.111 | 0.912 | 0.623 |

Random Forest attains the highest discrimination; XGBoost, at a low Youden
operating point, is a high-sensitivity / high-NPV rule-out candidate
(catching 17/18 metastatic nodes) but with correspondingly low
specificity. Wide confidence intervals reflect the small number of
metastatic observations (n = 18); external validation on an independent
cohort is required before any clinical translation.

Fourteen of 49 folds relied on a top-10 raw-p-value fallback because
fewer than three features survived FDR on the fold's training split — a
limitation imposed by the small positive class.

## Sensitivity analysis: repeated 5-fold StratifiedGroupKFold (20 repeats)

To complement the LOOCV point estimates, the same per-fold pipeline
(selection → scaling → SMOTE → fit) was run inside 20 independent
repeats of 5-fold StratifiedGroupKFold (groups = patient, stratified on
LN-level metastatic label). Per-held-out-fold AUCs (n = 99) are
summarised below alongside the LOOCV estimates:

| Model | LOOCV AUC (95% CI) | GroupKFold mean | GroupKFold median | GroupKFold 2.5–97.5% |
|---|---|---|---|---|
| Random Forest | 0.649 (0.474–0.831) | **0.670** | 0.702 | 0.293–0.964 |
| XGBoost | 0.631 (0.501–0.772) | 0.642 | 0.618 | 0.306–0.965 |
| Logistic Regression | 0.522 (0.322–0.746) | 0.560 | 0.597 | 0.188–0.963 |
| SVM | 0.443 (0.265–0.678) | 0.544 | 0.549 | 0.174–0.952 |

The two cross-validation schemes agree on the model ranking and on the
overall performance level. The GroupKFold percentile range is wider than
the LOOCV bootstrap CI — a direct consequence of the very small positive
class (18 metastatic observations across 49 patients): any single fold's
AUC is high-variance, regardless of the resampling scheme. The
fold-level fallback rate is also higher under GroupKFold (≈72%) than
LOOCV (≈29%) because each training split has fewer patients, so fewer
features survive FDR. Together, the two analyses give a more honest
picture of model performance than either does alone.

## Key Features

- Interactive DICOM viewer for CT / mask inspection
- Automated resampling to 1 mm isotropic voxels
- PyRadiomics feature extraction (107 features / node), with optional
  `pyradiomics-cuda` GPU extractor fallback
- Nested leave-one-patient-out cross-validation with per-fold feature
  selection, scaling, and SMOTE — all fit on training data only
- Four classifiers (LR, RF, SVM, XGBoost) with per-fold SMOTE and
  `class_weight="balanced"`
- Patient-level bootstrap 95% CIs on AUC (5,000 resamples, resampling
  patients rather than observations)
- Feature-selection stability tracking across folds
- NVIDIA CUDA acceleration for XGBoost when available (auto-detected)
- SHAP analysis, calibration curves, per-LN-site subgroup analysis

## Requirements

```
pip install SimpleITK pyradiomics scikit-learn imbalanced-learn xgboost \
            matplotlib ipywidgets pydicom numpy pandas scipy statsmodels
# Optional GPU path for feature extraction:
pip install pyradiomics-cuda
```

XGBoost wheels from PyPI include CUDA support on Windows; the notebook
auto-detects a working CUDA device via a tiny probe fit and falls back to
CPU if unavailable.

## Usage

1. Place CT DICOM directories in `C:/Radiomics_Projects/working/data/images/`
   (directories ending in `_CT`).
2. Place NIfTI masks in `.../data/mask/` (filenames matching patient IDs).
3. Run `radiomics.ipynb` cells sequentially: *Kernel → Restart & Run All*.

Primary outputs written to the working directory:

| File | Contents |
|---|---|
| `radiomics_features.csv` | Full 107-feature matrix per LN observation |
| `full_data_feature_selection.csv` | Diagnostic full-data Mann–Whitney / FDR table (supplementary) |
| `nested_loocv_oof_predictions.csv` | Per-observation out-of-fold probabilities for every model |
| `nested_loocv_summary.csv` | Headline AUC / sens / spec / PPV / NPV / CIs |
| `feature_selection_stability.csv` | Features selected in each of the 49 folds |
| `feature_selection_frequency.csv` | Aggregate selection frequency per feature |
| `repeated_groupkfold_per_fold.csv` | Per-fold AUCs from the GroupKFold sensitivity run |
| `repeated_groupkfold_summary.csv` | GroupKFold summary statistics per model |
| `auc_loocv_vs_groupkfold.csv` | Side-by-side LOOCV vs GroupKFold comparison |
| `nested_loocv_roc.png` | 4-panel ROC curves |
| `nested_loocv_confusion.png` | Confusion matrices at Youden threshold |
| `feature_selection_stability.png` | Top-15 feature selection frequency chart |
| `repeated_groupkfold_boxplot.png` | Per-fold AUC distribution per model (sensitivity analysis) |

## Citation

Pinard CJ et al. *CT-based radiomic features predict cervical lymph node
metastasis in dogs with oral malignancy: a machine-learning study using
leave-one-patient-out cross-validation.*

## License

MIT — see `LICENSE` file (add if needed).

## Contact

Christopher J. Pinard — christopher.pinard@animl.health
