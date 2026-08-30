"""
Preprocessing utilities: handling of biologically-impossible zero values
(treated as missing values) and outlier detection / capping (winsorizing)
based on the IQR method.
"""

import numpy as np
import pandas as pd

# Columns where a value of 0 is not physiologically possible and should
# therefore be treated as a missing value.
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def zeros_to_missing(dataframe: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Replace biologically-impossible zeros with NaN, then impute them with
    the median of each column, computed per target class (Outcome).

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str], optional
        Columns to treat. Defaults to ZERO_AS_MISSING_COLS.

    Returns
    -------
    pd.DataFrame
        Dataframe with zeros replaced and missing values imputed.
    """
    columns = columns or ZERO_AS_MISSING_COLS
    dataframe = dataframe.copy()

    for col in columns:
        n_zero = dataframe.loc[dataframe[col] == 0].shape[0]
        print(f"{col}: {n_zero} zero values converted to NaN")
        dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])

    dataframe = dataframe.fillna(dataframe.groupby("Outcome").transform("median"))
    return dataframe


def compute_global_medians(dataframe: pd.DataFrame, columns=None) -> dict:
    """Freeze a global (label-free) per-column median, meant to be computed
    ONCE on a trusted reference population (e.g. the baseline) and reused
    later by `zeros_to_missing_infer()` on a *different* dataframe (a live
    batch) -- never recomputed from the batch itself.

    Unlike `zeros_to_missing`, this never touches or requires a label
    column: the median is global per column, not conditioned on Outcome.

    Parameters
    ----------
    dataframe : pd.DataFrame
    columns : list[str], optional
        Columns to treat. Defaults to ZERO_AS_MISSING_COLS.

    Returns
    -------
    dict[str, float]
        Column -> frozen median (computed after treating 0 as missing).
    """
    columns = columns or ZERO_AS_MISSING_COLS
    medians = {}
    for col in columns:
        if col not in dataframe.columns:
            continue
        series = dataframe[col].replace(0, np.nan)
        medians[col] = float(series.median())
    return medians


def zeros_to_missing_infer(dataframe: pd.DataFrame, medians: dict, columns=None) -> pd.DataFrame:
    """Inference-safe counterpart of `zeros_to_missing()`.

    Replaces biologically-impossible zeros with NaN, then imputes them
    with medians FROZEN elsewhere (`compute_global_medians()`, typically
    on the baseline) -- never recomputed from `dataframe` itself, and
    never conditioned on a label column, which by definition is unknown
    at scoring time. This is what makes single-batch inference safe: the
    circular dependency of `zeros_to_missing()` (needs `Outcome` to
    impute, but `Outcome` is exactly what's being predicted) does not
    exist here.

    Parameters
    ----------
    dataframe : pd.DataFrame
    medians : dict[str, float]
        Frozen medians, as produced by `compute_global_medians()`.
    columns : list[str], optional
        Columns to treat. Defaults to ZERO_AS_MISSING_COLS.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    KeyError
        If a column needing imputation has no frozen median available in
        `medians` -- surfaces the missing frozen stat loudly instead of
        silently leaving NaNs in the scored batch.
    """
    columns = columns or ZERO_AS_MISSING_COLS
    dataframe = dataframe.copy()

    for col in columns:
        if col not in dataframe.columns:
            continue
        n_zero = dataframe.loc[dataframe[col] == 0].shape[0]
        print(f"{col}: {n_zero} zero values converted to NaN (frozen-median inference path)")
        dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])
        if dataframe[col].isna().any():
            if col not in medians:
                raise KeyError(
                    f"No frozen median available for '{col}' -- call "
                    f"compute_global_medians() on the reference population first."
                )
            dataframe[col] = dataframe[col].fillna(medians[col])

    return dataframe


def outlier_thresholds(dataframe: pd.DataFrame, variable: str,
                        low_quantile: float = 0.10, up_quantile: float = 0.90):
    """Compute lower/upper outlier thresholds for a variable using the IQR method.

    Returns
    -------
    (low_limit, up_limit) : tuple[float, float]
    """
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe: pd.DataFrame, col_name: str) -> bool:
    """Return True if the given column contains at least one outlier."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe: pd.DataFrame, variable: str) -> None:
    """Cap (winsorize) a column's values in place at its outlier thresholds."""
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # Cast to float first: recent pandas raises when assigning a float
    # threshold into an int64 column (e.g. Age, Pregnancies).
    if dataframe[variable].dtype.kind == "i":
        dataframe[variable] = dataframe[variable].astype("float64")
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit
