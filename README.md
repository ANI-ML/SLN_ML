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

Restricted to the 35 LOOCV folds where ≥3 features survived BH-FDR
(without the top-10 fallback), these same four features were selected in
35, 35, 34, and 32 of 35 folds respectively — so their stability is not
an artefact of the fallback rule.

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
fewer than three features survived FDR on the fold's training split
(see `fold_fallback_report.csv`) — a limitation imposed by the small
positive class.

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

## Cluster-robust sensitivity: GEE (exchangeable correlation)

Block D's Mann–Whitney U test treats every LN observation as
independent, which overstates evidence when the same patient contributes
up to four LN observations. A single-feature logistic GEE
(`statsmodels`, exchangeable working correlation, `groups = patient_id`)
was refit for each of the 60 features evaluated in Block D, followed by
BH-FDR correction.

- 21 / 60 features are FDR-significant under the cluster-robust model.
- All 4 of the Block-D MWU-FDR-significant features remain
  FDR-significant under GEE (adjusted-p in the range 3.5×10⁻¹³ to
  1.2×10⁻²), so the main feature shortlist is robust to within-patient
  correlation.

Full results in `gee_sensitivity_results.csv`.

## QC and supplementary analyses

| File | What it shows |
|---|---|
| `dicom_protocol_extract.csv` | Per-patient DICOM acquisition parameters (manufacturer, kVp, tube current, exposure, kernel, slice thickness, contrast). Cohort is predominantly 120 kVp on GE BrightSpeed with 0.625 mm slice thickness, but scanner and slice-thickness diversity (45 × 0.625 mm, 2 × 2.5 mm, 1 × 1 mm, 1 × 2 mm, 1 × 5 mm) is documented for the Methods section. |
| `roi_size_report.csv`, `roi_size_distribution.png` | Voxel count per LN ROI (post-resampling to 1 mm isotropic). 200 attempted observations; median 1,556 voxels, IQR 878–2,588, range 0–31,323. 5 observations have <50 voxels (all skipped by the extractor's `minimumROISize=3` guard). |
| `fold_fallback_report.csv`, `feature_frequency_nonfallback.csv` | Per-fold breakdown of how often the top-10 raw-p fallback was triggered (14/49 = 28.6%), and feature-selection frequency restricted to the 35 non-fallback folds. Top four features selected in 32–35 of 35 strict folds. |
| `per_fold_feature_importance.csv`, `feature_importance_aggregated.csv`, `per_fold_feature_importance.png` | Per-fold feature importances for Random Forest (impurity) and XGBoost (gain) across the 49 LOOCV folds, with min/max error bars on the top 10 mean importances. |
| `gee_sensitivity_results.csv` | Single-feature GEE models (exchangeable, patient clusters) — cluster-robust validation of the univariate effects. |
| `auc_consistency_check.csv` | Recomputes model AUCs from `nested_loocv_oof_predictions.csv` and asserts that they match `nested_loocv_summary.csv` within 1×10⁻³. A run that fails this check is not reportable. |
| `shap_oof_bar.png`, `shap_oof_beeswarm.png` | SHAP feature attribution for the top-4 features. *Attribution visualisation only — the SHAP model is refit on the full data set so its AUC is optimistically biased and must not be quoted. Honest performance is the nested-LOOCV table above.* |
| `roc_curves_BLOCK_E_supplementary.png` etc. | 70/30 patient-level split results retained strictly as a supplementary comparison against the nested-LOOCV pipeline. The 70/30 block is flagged as optimistic (feature selection used the full cohort). |

## Key Features

- Interactive DICOM viewer for CT / mask inspection
- Automated resampling to 1 mm isotropic voxels
- PyRadiomics feature extraction (107 features / node), with threaded
  CPU parallelism (`joblib`)
- Nested leave-one-patient-out cross-validation with per-fold feature
  selection, scaling, and SMOTE — all fit on training data only
- Four classifiers (LR, RF, SVM, XGBoost) with per-fold SMOTE and
  `class_weight="balanced"`
- Patient-level bootstrap 95% CIs on AUC (5,000 resamples, resampling
  patients rather than observations)
- Feature-selection stability tracking across folds + fallback reporting
- Per-fold feature-importance capture (RF impurity, XGBoost gain)
- Cluster-robust GEE sensitivity analysis (exchangeable correlation,
  patient clusters)
- DICOM acquisition-protocol audit and per-ROI voxel-count QC
- NVIDIA CUDA acceleration for XGBoost when available (auto-detected)
- SHAP feature-attribution plots (clearly flagged as optimistic refit)
- Automatic consistency check between the summary table and the OOF
  predictions (hard-fails on >1×10⁻³ drift)

## Requirements

```
pip install SimpleITK pyradiomics scikit-learn imbalanced-learn xgboost \
            matplotlib ipywidgets pydicom numpy pandas scipy statsmodels shap
```

XGBoost wheels from PyPI include CUDA support on Windows; the notebook
auto-detects a working CUDA device via a tiny probe fit and falls back
to CPU if unavailable.

## Usage

1. Place CT DICOM directories in `ct_working_data/ct_images/`
   (directories ending in `_CT`).
2. Place NIfTI masks in `ct_working_data/ct_mask/` (filenames matching
   patient IDs).
3. Run `radiomics.ipynb` cells sequentially: *Kernel → Restart & Run All*.

Primary outputs written to the working directory:

| File | Contents |
|---|---|
| `radiomics_features.csv` | Full 107-feature matrix per LN observation |
| `dicom_protocol_extract.csv` | Per-patient CT acquisition parameters (Block B2) |
| `roi_size_report.csv`, `roi_size_distribution.png` | Per-LN voxel counts and histogram (Block C) |
| `full_data_feature_selection.csv` | Diagnostic full-data Mann–Whitney / FDR table (supplementary) |
| `gee_sensitivity_results.csv` | Cluster-robust GEE single-feature sensitivity (Block D2) |
| `nested_loocv_oof_predictions.csv` | Per-observation out-of-fold probabilities for every model |
| `nested_loocv_summary.csv` | Headline AUC / sens / spec / PPV / NPV / CIs |
| `feature_selection_stability.csv` | Features selected in each of the 49 folds |
| `feature_selection_frequency.csv` | Aggregate selection frequency per feature |
| `fold_fallback_report.csv`, `feature_frequency_nonfallback.csv` | Fallback-rule usage breakdown + strict-fold feature frequency |
| `per_fold_feature_importance.csv`, `feature_importance_aggregated.csv` | Per-fold RF / XGBoost feature importance (Block F) |
| `repeated_groupkfold_per_fold.csv` | Per-fold AUCs from the GroupKFold sensitivity run |
| `repeated_groupkfold_summary.csv` | GroupKFold summary statistics per model |
| `auc_loocv_vs_groupkfold.csv` | Side-by-side LOOCV vs GroupKFold comparison |
| `auc_consistency_check.csv` | Assertion that summary and OOF predictions agree within 1×10⁻³ |
| `nested_loocv_roc.png` | 4-panel ROC curves |
| `nested_loocv_confusion.png` | Confusion matrices at Youden threshold |
| `feature_selection_stability.png` | Top-15 feature selection frequency chart |
| `per_fold_feature_importance.png` | Top-10 per-fold importances with min/max error bars |
| `repeated_groupkfold_boxplot.png` | Per-fold AUC distribution per model (sensitivity analysis) |
| `shap_oof_bar.png`, `shap_oof_beeswarm.png` | SHAP attribution plots (optimistic refit; attribution only) |

## Citation

Pinard CJ et al. *CT-based radiomic features predict cervical lymph node
metastasis in dogs with oral malignancy: a machine-learning study using
leave-one-patient-out cross-validation.*

## License

MIT — see `LICENSE` file (add if needed).

## Contact

Christopher J. Pinard — christopher.pinard@animl.health
