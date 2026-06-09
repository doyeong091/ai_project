# back/server/model_router.py

from __future__ import annotations

from typing import Any

SHOCK_KEYWORDS = [
    "hormuz",
    "strait_of_hormuz",
    "gulf_supply",
    "supply_disruption",
    "oil_infrastructure_attack",
    "infrastructure_attack",
    "pipeline_attack",
    "refinery_attack",
    "gdelt_global_hormuz_risk_count",
    "gdelt_middleeast_hormuz_risk_count",
    "gdelt_global_gulf_supply_disruption_count",
    "gdelt_middleeast_gulf_supply_disruption_count",
    "gdelt_global_oil_infrastructure_attack_count",
    "gdelt_middleeast_oil_infrastructure_attack_count",
]


def normalize_feature_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def is_shock_related_feature(feature_name: str) -> bool:
    normalized = normalize_feature_name(feature_name)

    return any(keyword in normalized for keyword in SHOCK_KEYWORDS)


def is_meaningful_value(value: Any) -> bool:
    """
    shock feature가 존재하더라도 값이 0, None, 빈 문자열이면
    실제 shock 입력으로 보지 않는다.
    """
    if value is None:
        return False

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return False

    try:
        numeric_value = float(value)

        return abs(numeric_value) > 1e-9

    except Exception:
        return True


def should_use_shock_aware_model(selected_features: dict[str, Any] | None) -> bool:
    if not selected_features:
        return False

    for feature_name, value in selected_features.items():
        if not is_shock_related_feature(feature_name):
            continue

        if is_meaningful_value(value):
            return True

    return False


def route_model_type(selected_features: dict[str, Any] | None) -> str:
    if should_use_shock_aware_model(selected_features):
        return "shock_aware"

    return "default"
