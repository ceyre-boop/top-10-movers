"""T1/T2 feature builders. See docs/LABEL_SPEC.md and plan §4."""

from top10.features.spec import (
    T1_SPEC,
    T2_SPEC,
    FeatureSpec,
    feature_output_path,
    validate_frame,
    write_features,
)
from top10.features.t1 import build_t1_features, decision_time_t1
from top10.features.t2 import build_t2_features, decision_time_t2

__all__ = [
    "FeatureSpec",
    "T1_SPEC",
    "T2_SPEC",
    "validate_frame",
    "feature_output_path",
    "write_features",
    "build_t1_features",
    "decision_time_t1",
    "build_t2_features",
    "decision_time_t2",
]
