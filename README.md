# CT Radiomics for Canine LN Metastasis Prediction

This repository contains the code, data processing pipeline, and analysis for the manuscript "CT-based radiomic features predict cervical lymph node metastasis in dogs with oral malignancy: a machine learning study using leave-one-patient-out cross-validation."

## Overview
The project analyzes CT images from 49 dogs with oral tumors, segmenting bilateral mandibular and retropharyngeal lymph nodes (195 observations, 18 metastatic). PyRadiomics extracts 107 features per node, reduced to 4 key texture features (GLCM-MCC, GLSZM SmallAreaLowGrayLevelEmphasis, GLSZM LowGrayLevelZoneEmphasis, GLSZM HighGrayLevelZoneEmphasis) via FDR-corrected selection. 

XGBoost achieves LOOCV AUC 0.793 (95% CI 0.627-0.895), with high NPV (95.8%) for ruling out metastasis. 

## Key Features
- Interactive DICOM viewer for CT/mask inspection
- Automated resampling to 1mm isotropic voxels
- PyRadiomics feature extraction
- ML models (XGBoost, RF, LR, SVM) with SMOTE and patient-level LOOCV
- Bootstrap CIs, SHAP analysis, feature importance

## Requirements
```bash
pip install SimpleITK pyradiomics scikit-learn imbalanced-learn xgboost matplotlib ipywidgets pydicom numpy pandas
```

## Usage
1. Place CT DICOM directories in `C:/Radiomics_Projects/working/data/images/` (ending in `_CT`)
2. Place NIfTI masks in `.../data/mask/` (matching patient IDs)
3. Run `radiomics.ipynb` cells sequentially for processing/visualization/ML

Expected outputs: `mlreadyfeatures.csv`, ROC/confusion matrices, feature importance plots.

## Results Summary
| Model | LOOCV AUC | Sensitivity | Specificity | NPV |
|-------|-----------|-------------|-------------|-----|
| XGBoost | 0.793 | 0.611 | 0.910 | 0.958% 
| Random Forest | 0.760 | 0.500 | 0.898 | 0.946 |

## Citation
Pinard CJ et al. CT-based radiomic features predict cervical lymph node metastasis in dogs with oral malignancy.

## License
MIT - see LICENSE file (add if needed).

## Contact
Christopher J. Pinard (christopher.pinard@animl.health)
