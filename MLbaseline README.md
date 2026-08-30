# Diabetes Prediction — Pima Indians Diabetes Database

Binary classification project predicting whether a patient has diabetes from 8 diagnostic
measurements, using the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Project structure

```
diabetes-prediction/
├── data/                              # raw & intermediate data (gitignored except .gitkeep)
├── models/                            # trained model artifacts (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                       # first structured look at the raw data
│   ├── 02_preprocessing.ipynb             # missing values (zero -> NaN) + outlier capping
│   ├── 03_feature_engineering_visualization.ipynb  # clinical feature engineering + plots
│   ├── 04_modeling.ipynb                  # baselines, tuning, voting ensemble
│   └── 05_evaluation.ipynb                # test-set metrics, ROC, confusion matrix (notebook version)
├── src/
│   ├── eda.py                         # check_df, grab_col_names, check_missing_value
│   ├── preprocessing.py               # zeros_to_missing, outlier handling
│   ├── feature_engineering.py         # feature_extraction, encoders, data_prep pipeline
│   ├── visualization.py               # all plotting helpers
│   └── modeling.py                    # base_models, hyperparameter_optimization, voting_classifier
├── evaluate.py                        # terminal evaluation report (run with `python evaluate.py`)
├── requirements.txt
└── README.md
```

Reusable logic lives in `src/` as plain, tested-in-notebook functions; the notebooks stay
focused on narrative, exploration, and results — nothing gets copy-pasted between them.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `diabetes.csv` from Kaggle and place it at `data/diabetes.csv`, then run the
notebooks in order (01 → 05). Each notebook persists its output (`data/diabetes_cleaned.csv`,
`data/diabetes_features.csv`, `models/*.pkl`) so the next one can pick up from there without
recomputation.

## Pipeline summary

1. **EDA** — shape, types, missing values, column classification.
2. **Preprocessing** — biologically-impossible zeros (`Glucose`, `BloodPressure`,
   `SkinThickness`, `Insulin`, `BMI`) are treated as missing and imputed with the per-class
   median; outliers are capped with the IQR method.
3. **Feature engineering** — clinical threshold-based categories (glucose tolerance,
   blood pressure bands, BMI bands, age groups, combined risk flag) + full EDA visualizations.
4. **Modeling** — baseline comparison before/after preprocessing (LR, KNN, CART, RF, GBM,
   XGBoost, LightGBM), `GridSearchCV` tuning, and a soft-voting ensemble.
5. **Evaluation** — classification report, ROC curve, confusion matrix, feature importance
   on a held-out test set.

## Try it: terminal evaluation report

`notebooks/04_modeling.ipynb` saves the trained model and the held-out test set
(`models/lgbm_model.pkl`, `data/X_test.csv`, `data/y_test.csv`). Once that notebook has run,
print a full evaluation report straight to the terminal:

```bash
python evaluate.py
```

```
============================================================
 MODEL EVALUATION — models/lgbm_model.pkl
 Test set: 154 patients
============================================================

--- Summary metrics ---
Accuracy  : 0.9026
Precision : 0.8730
Recall    : 0.8302
F1 Score  : 0.8511
ROC AUC   : 0.9457

--- Classification report ---
              precision    recall  f1-score   support
Non-diabetic       0.92      0.94      0.93       101
    Diabetic       0.87      0.83      0.85        53
...

--- Confusion matrix ---
                 Predicted
               Neg     Pos
Actual  Neg      95       6
        Pos       9      44

--- Top feature importances ---
  Glucose                         ##############################  312
  BMI                              #########################       260
  ...
```

To evaluate the voting ensemble instead: `python evaluate.py --model models/voting_clf.pkl`.

## Results

| Algorithm | Baseline (raw) | Baseline (preprocessed) | Tuned |
|---|---|---|---|
| LR | 0.7671 | 0.8731 | - |
| KNN | 0.7182 | 0.8535 | 0.8616 |
| CART | 0.7312 | 0.8354 | 0.8615 |
| RF | 0.7752 | 0.8893 | 0.8828 |
| GBM | 0.7638 | 0.8893 | - |
| XGBoost | 0.7508 | 0.8909 | 0.8974 |
| **LightGBM** | 0.7378 | 0.8925 | **0.9007** |

LightGBM gives the best accuracy after tuning. `Glucose`, `Insulin`, and `BMI` are the
strongest predictors, and preprocessing has a bigger impact on accuracy than model choice.

## License

Dataset from the National Institute of Diabetes and Digestive and Kidney Diseases, released
under the CC0 license on Kaggle.
