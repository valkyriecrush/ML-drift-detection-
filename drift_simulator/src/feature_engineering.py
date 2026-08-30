"""
Feature engineering utilities: domain-knowledge-driven categorical features
(clinical thresholds for Insulin, BloodPressure, Glucose, BMI, Age), encoding
helpers, and the full train/test preprocessing pipeline used before modeling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.eda import grab_col_names
from src.preprocessing import ZERO_AS_MISSING_COLS, replace_with_thresholds


def feature_extraction(dataframe: pd.DataFrame) -> None:
    """Add clinically-motivated categorical features in place:

    - Insulin_CAT, Glucose_CAT: Normal / Prediabetes / Diabetes bands
    - BloodPressure_CAT: standard hypertension bands
    - BMI_CAT: standard WHO BMI bands
    - Age_CAT: Adult / Middle_Age_Adult / Senior_Adult
    - Life_Level_CAT: combined "at risk" flag (Age > 40, hypertensive,
      overweight or obese)
    """
    dataframe["Insulin_CAT"] = pd.cut(
        x=dataframe["Insulin"], bins=[0, 140, 199, np.inf],
        labels=["Normal", "Prediabetes", "Diabetes"])

    dataframe["BloodPressure_CAT"] = pd.cut(
        x=dataframe["BloodPressure"], bins=[0, 60, 80, 90, 120, np.inf],
        labels=["Low_Blood_Pressure", "Normal", "Prehypertension",
                "Hypertension", "Hypertensive_Crisis"])

    dataframe["Glucose_CAT"] = pd.cut(
        x=dataframe["Glucose"], bins=[0, 140, 199, np.inf],
        labels=["Normal", "Impaired_Glucose_Tolerance", "Diabetes"])

    dataframe["BMI_CAT"] = pd.cut(
        x=dataframe["BMI"], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf],
        labels=["Underweight", "Healthy", "Overweight",
                "Obese_Class1", "Obese_Class2", "Obese_Class3"])

    dataframe["Age_CAT"] = pd.cut(
        x=dataframe["Age"], bins=[20, 40, 60, np.inf],
        labels=["Adult", "Middle_Age_Adult", "Senior_Adult"])

    dataframe.loc[
        (dataframe["Age"] > 40)
        & (dataframe["BloodPressure_CAT"] == "Hypertension")
        & (dataframe["BMI_CAT"].isin(["Overweight", "Obese_Class1", "Obese_Class2", "Obese_Class3"])),
        "Life_Level_CAT"
    ] = "At_Risk"

    # NOTE: `dataframe["Life_Level_CAT"].fillna(..., inplace=True)` silently
    # no-ops on modern pandas (Copy-on-Write applies to the temporary Series
    # returned by __getitem__, not to `dataframe`). Reassign explicitly.
    dataframe["Life_Level_CAT"] = dataframe["Life_Level_CAT"].fillna("Not_Risk")


def one_hot_encoder(dataframe: pd.DataFrame, categorical_col, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode the given categorical columns."""
    return pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first)


def label_encoder(dataframe: pd.DataFrame, binary_col: str) -> pd.DataFrame:
    """Label-encode a single binary column in place."""
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe


def data_prep(X: pd.DataFrame, y: pd.Series):
    """Full preprocessing pipeline applied independently to a train or test
    split: outlier capping, zero->missing imputation, feature extraction,
    robust scaling of numeric columns, and encoding of categorical columns.

    Fitting this per-split (rather than on the full dataset before the
    train/test split) avoids data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector ("Outcome").

    Returns
    -------
    X : pd.DataFrame
        Fully preprocessed feature matrix.
    y : pd.Series
        Target vector, realigned with X's index.
    """
    index = X.index
    dataframe = X.merge(y.to_frame(), left_index=True, right_index=True).set_index(index)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, print_results=False)

    # Outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # Zero -> missing, then median imputation per class
    for col in ZERO_AS_MISSING_COLS:
        dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])
    dataframe = dataframe.fillna(dataframe.groupby("Outcome").transform("median"))

    # Feature engineering
    feature_extraction(dataframe)

    # Scaling
    rs = RobustScaler()
    dataframe[num_cols] = rs.fit_transform(dataframe[num_cols])

    # Binary encoding
    binary_cols = [col for col in dataframe.columns
                   if dataframe[col].dtype not in ["int64", "float64"]
                   and dataframe[col].nunique() == 2]
    for col in binary_cols:
        label_encoder(dataframe, col)

    # One-hot encoding
    ohe_cols = [col for col in dataframe.columns if 12 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)

    X = dataframe.drop(["Outcome"], axis=1)
    y = dataframe["Outcome"]
    return X, y
