# Purpose: demonstrates the feature store end-to-end — historical features for a batch, and online features for one new patient, fed straight into the already-trained model.

from __future__ import annotations

import sys

import joblib

from feature_store.store import FeatureStore


def main() -> None:
    store = FeatureStore().materialize()
    print(store.describe())

    print("\n=== Historical features (first 3 rows, unscaled) ===")
    hist = store.get_historical_features()
    print(hist.head(3).to_string(index=False))

    model_path = "models/lgbm_model.pkl"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"\n[skip] {model_path} not found — run the notebooks first to train a model.")
        sys.exit(0)

    # A brand-new patient, never seen by the store or the model — only the
    # 8 raw clinical measurements are required.
    new_patient = {
        "Pregnancies": 2,
        "Glucose": 150,
        "BloodPressure": 78,
        "SkinThickness": 32,
        "Insulin": 0,       # 0 -> biologically-impossible -> imputed like training
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.52,
        "Age": 45,
    }

    print("\n=== Online features for one new patient ===")
    online_row = store.get_online_features(new_patient)
    print(online_row.to_string(index=False))

    proba = model.predict_proba(online_row)[0, 1]
    print(f"\nPredicted diabetes probability: {proba:.4f}")


if __name__ == "__main__":
    main()
