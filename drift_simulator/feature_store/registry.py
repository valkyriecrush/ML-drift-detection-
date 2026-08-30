# Purpose: declarative catalog of feature groups, loaded from config/feature_store.yml (what features exist, grouped how, computed from what).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureGroup:
    """One named, versionable set of features that are always produced /
    retrieved together (the "feature view" of most feature stores)."""

    name: str
    description: str
    features: List[str]
    entity: str
    online_safe: bool = True
    depends_on: Optional[str] = None


@dataclass
class FeatureRegistry:
    """Read-only view over `feature_store:` in config/feature_store.yml.

    Kept separate from FeatureStore (which owns materialization / I/O) so
    the catalog of "what features exist" can be inspected or validated
    without touching disk caches or fitted artifacts.
    """

    config: Dict[str, Any]
    entity_key: str = field(init=False)
    target_col: str = field(init=False)
    groups: Dict[str, FeatureGroup] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        fs = self.config["feature_store"]
        self.entity_key = fs["entity_key"]
        self.target_col = fs["source"]["target_col"]

        self.groups = {}
        for name, spec in fs["feature_groups"].items():
            self.groups[name] = FeatureGroup(
                name=name,
                description=" ".join(spec.get("description", "").split()),
                features=list(spec["features"]),
                entity=spec.get("entity", self.entity_key),
                online_safe=bool(spec.get("online_safe", True)),
                depends_on=spec.get("depends_on"),
            )

    def list_groups(self) -> List[str]:
        return list(self.groups.keys())

    def get_group(self, name: str) -> FeatureGroup:
        if name not in self.groups:
            raise KeyError(
                f"Unknown feature group '{name}'. Available: {self.list_groups()}"
            )
        return self.groups[name]

    def all_feature_names(self) -> List[str]:
        names: List[str] = []
        for group in self.groups.values():
            names.extend(group.features)
        return names

    def raw_numeric_features(self) -> List[str]:
        """The 8 original clinical measurements (raw_measurements group)."""
        return list(self.groups["raw_measurements"].features)

    def online_safe_groups(self) -> List[str]:
        """Groups that can be computed for a single new record without a
        statistical dependency beyond persisted fit-time artifacts."""
        return [g.name for g in self.groups.values() if g.online_safe]

    def describe(self) -> str:
        lines = [
            f"Feature registry: {len(self.groups)} group(s), "
            f"{len(self.all_feature_names())} feature(s), entity='{self.entity_key}'"
        ]
        for group in self.groups.values():
            lines.append(f"  - {group.name} ({len(group.features)} features): {group.description}")
        return "\n".join(lines)
