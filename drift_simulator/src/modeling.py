"""
Modeling utilities: baseline model comparison, hyperparameter optimization
(grid search) and a soft-voting ensemble.
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Hyperparameter search space for each tuned model.
HYPERPARAM_GRID = {
    "KNN": {"n_neighbors": range(2, 50)},
    "CART": {"max_depth": range(1, 20), "min_samples_split": range(2, 30)},
    "RF": {"max_depth": [8, 15, None], "max_features": [5, 7],
           "min_samples_split": [15, 20], "n_estimators": [200, 300]},
    "XGBoost": {"learning_rate": [0.1, 0.01], "max_depth": [5, 8],
                "n_estimators": [100, 200], "colsample_bytree": [0.5, 1]},
    "LightGBM": {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500, 1500],
                 "colsample_bytree": [0.5, 0.7, 1]},
}


def get_base_classifiers():
    """Return the list of (name, estimator) baseline classifiers."""
    return [
        ("LR", LogisticRegression(solver="lbfgs", max_iter=3000)),
        ("KNN", KNeighborsClassifier()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ("GBM", GradientBoostingClassifier()),
        ("XGBoost", XGBClassifier(eval_metric="logloss")),
        ("LightGBM", LGBMClassifier()),
    ]


def base_models(X, y, scoring="accuracy", cv=5):
    """Cross-validate a set of baseline classifiers and print their scores."""
    print(f"Base Models ({scoring})....")
    for name, classifier in get_base_classifiers():
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name})")


def hyperparameter_optimization(X, y, cv=5, scoring="accuracy"):
    """Grid-search-tune KNN, CART, RF, XGBoost and LightGBM. Returns a dict
    of {name: fitted-with-best-params estimator}."""
    tunable_classifiers = [
        ("KNN", KNeighborsClassifier(), HYPERPARAM_GRID["KNN"]),
        ("CART", DecisionTreeClassifier(), HYPERPARAM_GRID["CART"]),
        ("RF", RandomForestClassifier(), HYPERPARAM_GRID["RF"]),
        ("XGBoost", XGBClassifier(eval_metric="logloss"), HYPERPARAM_GRID["XGBoost"]),
        ("LightGBM", LGBMClassifier(), HYPERPARAM_GRID["LightGBM"]),
    ]

    print("\nHyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in tunable_classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model
    return best_models


def voting_classifier(best_models, X, y):
    """Fit a soft-voting ensemble of KNN + RF + LightGBM and report its
    cross-validated accuracy / F1 / ROC AUC."""
    print("\nVoting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[("KNN", best_models["KNN"]),
                    ("RF", best_models["RF"]),
                    ("LightGBM", best_models["LightGBM"])],
        voting="soft"
    ).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1 Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC : {cv_results['test_roc_auc'].mean()}")
    return voting_clf


def fit_models(X, y):
    """Full modeling pipeline: baseline comparison -> hyperparameter tuning
    -> voting ensemble. Returns (voting_clf, best_models)."""
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    return voting_clf, best_models
