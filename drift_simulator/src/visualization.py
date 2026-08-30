"""
Plotting utilities for the EDA and model-evaluation notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(dataframe: pd.DataFrame, num_cols) -> None:
    """Plot a correlation heatmap for the given numeric columns."""
    corr = dataframe[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="Blues", annot=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def boxplot_for_outliers(dataframe: pd.DataFrame, num_cols, title: str = None) -> None:
    """Plot one boxplot per numeric column, arranged in a 2x4 grid."""
    a, b, c = 2, 4, 1
    palette = sns.color_palette("Set3", len(num_cols)).as_hex()

    fig = plt.figure(figsize=(20, 10))
    for col, color in zip(num_cols, palette):
        plt.subplot(a, b, c)
        sns.boxplot(x=dataframe[col], color=color)
        plt.xlabel(f"{dataframe[col].name}", size=15)
        c += 1
    plt.suptitle(title or "Boxplots", size=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.show()


def hist_plot(dataframe: pd.DataFrame, columns, title: str) -> None:
    """Plot histograms (with KDE) for the given columns, arranged in a 4x2 grid."""
    a, b, c = 4, 2, 1
    palette = sns.color_palette("Paired", len(columns)).as_hex()

    fig = plt.figure(figsize=(13, 15))
    for col, color in zip(columns, palette):
        plt.subplot(a, b, c)
        sns.histplot(data=dataframe, x=col, kde=True, color=color)
        plt.xlabel(f"{dataframe[col].name}", size=15)
        c += 1
    plt.suptitle(title, size=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.show()


def plot_target_distribution(dataframe: pd.DataFrame, target_col: str = "Outcome") -> None:
    """Plot the count and pie-chart ratio of the (binary) target variable."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    ax = sns.countplot(
        x=dataframe[target_col].replace({0: "NEGATIVE", 1: "POSITIVE"}),
        ax=axs[0], palette=["#ffcce7", "#81b7d2"])
    ax.set_xlabel("Outcome", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    axs[0].set_title("Count of Diabetes", fontsize=16)

    def func(pct, allvals):
        absolute = int(np.round(pct / 100. * np.sum(allvals)))
        return f"{pct:.2f}%\n({absolute:d})"

    counts = dataframe[target_col].value_counts()
    axs[1].pie(
        counts, explode=[0, 0.07], colors=["#81b7d2", "#ffcce7"],
        autopct=lambda pct: func(pct, counts), labels=["NEGATIVE", "POSITIVE"],
        textprops=dict(color="black", size=13))
    axs[1].set_title("Ratio of Diabetes", fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_categorical_features_vs_target(dataframe: pd.DataFrame, new_cols, target_col: str = "Outcome") -> None:
    """Plot count of each engineered categorical feature, split by target."""
    a, b, c = 2, 2, 1
    fig = plt.figure(figsize=(15, 11))
    for col in new_cols:
        plt.subplot(a, b, c)
        sns.countplot(x=dataframe[col], hue=dataframe[target_col],
                      palette=["#AA96DA", "#C5FAD5"])
        plt.ylabel("Count")
        plt.xlabel(f"{col}", size=15)
        c += 1
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.17)
    plt.show()


def plot_importance(model, features, num=None) -> None:
    """Plot the top `num` feature importances of a fitted tree-based model."""
    num = num or len(features.columns)
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value", ascending=False)[:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
