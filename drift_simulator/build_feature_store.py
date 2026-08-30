# Purpose: CLI entry point that materializes the feature store (offline table + fitted online-serving artifacts).

from __future__ import annotations

import argparse

from feature_store.store import FeatureStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/feature_store.yml",
        help="Path to the feature store config (single source of truth for feature groups/paths).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute even if a cached offline table already matches the source data's hash.",
    )
    args = parser.parse_args()

    store = FeatureStore(config_path=args.config)
    store.materialize(force_recompute=args.force_recompute)

    print(store.describe())
    print(f"\nOffline table -> {store._offline_paths()['features']}")
    print(f"Online-serving artifacts -> {store._artifacts_path()}")


if __name__ == "__main__":
    main()
