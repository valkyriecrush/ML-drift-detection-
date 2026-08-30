"""
Exploratory Data Analysis utilities.

Functions used to get a first, structured overview of the raw dataset
(shape, types, missing values, duplicates, distribution) and to
automatically classify columns as categorical / numerical / cardinal.
"""

import pandas as pd


def check_df(data: pd.DataFrame, head: int = 5) -> None:
    """Print a structured summary of a dataframe: shape, dtypes, head/tail,
    a random sample, missing values, duplicates, unique values and
    descriptive statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to inspect.
    head : int, default=5
        Number of rows to show for head/tail/sample.
    """
    print("\n****** Shape ******")
    print(f"Shape     : {data.shape}\n"
          f"Size      : {data.size}\n"
          f"Dimension : {data.ndim}")

    print("\n****** Types ******")
    print(data.dtypes)

    print("\n****** Head ******")
    print(data.head(head))

    print("\n****** Tail ******")
    print(data.tail(head))

    print("\n****** Random Sampling ******")
    print(data.sample(head))

    print("\n****** Missing Values ******")
    print(data.isnull().sum())

    print("\n****** Duplicated Values ******")
    print(data.duplicated().sum())

    print("\n****** Unique Values ******")
    print(data.nunique())

    print("\n****** Describe ******")
    print(data.describe().T)


def grab_col_names(dataframe: pd.DataFrame, cat_th: int = 10, car_th: int = 20,
                    print_results: bool = True):
    """Classify the columns of a dataframe into categorical, numerical and
    "categorical but cardinal" (high-cardinality categorical) variables.

    Note: numerical columns with a low number of unique values are treated
    as categorical (e.g. a 0/1 flag stored as int).

    Parameters
    ----------
    dataframe : pd.DataFrame
    cat_th : int, optional
        Threshold under which a numeric column is considered categorical.
    car_th : int, optional
        Threshold above which a categorical column is considered cardinal.
    print_results : bool, default=True
        Whether to print a short report.

    Returns
    -------
    cat_cols : list[str]
    num_cols : list[str]
    cat_but_car : list[str]

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of columns
    """
    cat_cols = [col for col in dataframe.columns
                if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns
                   if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in dataframe.columns
                   if dataframe[col].nunique() > car_th
                   and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    if print_results:
        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables:    {dataframe.shape[1]}")
        print(f"cat_cols:     {len(cat_cols)}")
        print(f"num_cols:     {len(num_cols)}")
        print(f"cat_but_car:  {len(cat_but_car)}")
        print(f"num_but_cat:  {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def check_missing_value(dataframe: pd.DataFrame, na_name: bool = False):
    """Print a table of missing values (count + ratio) per column.

    Parameters
    ----------
    dataframe : pd.DataFrame
    na_name : bool, default=False
        If True, return the list of column names that contain missing values.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio.round(2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
