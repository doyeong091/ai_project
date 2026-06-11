# model/plot_model_tracking.py

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Path settings
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "back" / "models" / "final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_tracking"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = MODEL_DIR / "final_model_manifest.json"
FEATURES_PATH = MODEL_DIR / "final_model_features.json"

DATASET_PATHS = {
    "dubai": PROCESSED_DIR / "dubai_dataset.csv",
    "wti": PROCESSED_DIR / "wti_dataset.csv",
    "brent": PROCESSED_DIR / "brent_dataset.csv",
}

MODEL_PATHS = {
    "dubai": {
        "default": MODEL_DIR / "dubai_default_model.pkl",
        "shock_aware": MODEL_DIR / "dubai_shock_aware_model.pkl",
    },
    "wti": {
        "default": MODEL_DIR / "wti_default_model.pkl",
        "shock_aware": MODEL_DIR / "wti_shock_aware_model.pkl",
    },
    "brent": {
        "default": MODEL_DIR / "brent_default_model.pkl",
        "shock_aware": MODEL_DIR / "brent_shock_aware_model.pkl",
    },
}

CURRENT_PRICE_COLS = {
    "dubai": "current_Dubai",
    "wti": "current_WTI",
    "brent": "current_Brent",
}

OIL_DISPLAY_NAMES = {
    "dubai": "DUBAI",
    "wti": "WTI",
    "brent": "BRENT",
}

MODEL_TYPES = ["default", "shock_aware"]
OIL_TYPES = ["dubai", "wti", "brent"]

DEFAULT_TEST_START = pd.Timestamp("2025-01-01")
DEFAULT_TEST_END = pd.Timestamp("2025-12-31")


# ============================================================
# Load helpers
# ============================================================
def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def get_model_file_name(oil: str, model_type: str) -> str:
    return f"{oil}_{model_type}_model.pkl"


def get_feature_info(
    features_json: dict[str, Any],
    oil: str,
    model_type: str,
) -> dict[str, Any]:
    model_file_name = get_model_file_name(oil, model_type)

    if model_file_name in features_json:
        info = features_json[model_file_name]

        if isinstance(info, dict):
            return info

    oil_block = features_json.get(oil)

    if isinstance(oil_block, dict):
        mode_block = oil_block.get(model_type)

        if isinstance(mode_block, dict):
            return mode_block

        if isinstance(mode_block, list):
            return {"feature_cols": mode_block}

    for _, value in features_json.items():
        if not isinstance(value, dict):
            continue

        if value.get("oil_type") == oil and value.get("model_mode") == model_type:
            return value

    return {}


def get_feature_list(
    features_json: dict[str, Any],
    oil: str,
    model_type: str,
) -> list[str]:
    info = get_feature_info(features_json, oil, model_type)

    features = (
        info.get("feature_cols")
        or info.get("features")
        or info.get("selected_features")
        or info.get("input_features")
        or []
    )

    if not isinstance(features, list):
        return []

    return [str(feature) for feature in features]


def get_model_meta(
    manifest: dict[str, Any],
    features_json: dict[str, Any],
    oil: str,
    model_type: str,
) -> dict[str, Any]:
    model_file_name = get_model_file_name(oil, model_type)

    meta: dict[str, Any] = {}

    if isinstance(manifest.get(model_file_name), dict):
        meta.update(manifest[model_file_name])

    if isinstance(manifest.get(oil), dict) and isinstance(
        manifest[oil].get(model_type),
        dict,
    ):
        meta.update(manifest[oil][model_type])

    feature_info = get_feature_info(features_json, oil, model_type)
    meta.update({k: v for k, v in feature_info.items() if k != "feature_cols"})

    return meta


def extract_model_and_features(
    model_obj,
    fallback_features: list[str],
) -> tuple[Any, list[str]]:
    if isinstance(model_obj, dict):
        model = (
            model_obj.get("model")
            or model_obj.get("estimator")
            or model_obj.get("pipeline")
            or model_obj.get("best_model")
        )

        feature_cols = (
            model_obj.get("feature_cols")
            or model_obj.get("features")
            or model_obj.get("selected_features")
            or model_obj.get("input_features")
            or fallback_features
        )

        if model is None:
            raise ValueError(
                "pkl dict 안에서 model/estimator/pipeline/best_model 키를 찾지 못했습니다."
            )

        if not feature_cols and hasattr(model, "feature_names_in_"):
            feature_cols = list(model.feature_names_in_)

        if not feature_cols:
            raise ValueError(
                "모델 pkl과 final_model_features.json 양쪽에서 feature 목록을 찾지 못했습니다."
            )

        return model, list(feature_cols)

    model = model_obj
    feature_cols = list(fallback_features or [])

    if not feature_cols and hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)

    if not feature_cols:
        raise ValueError(
            "모델 pkl과 final_model_features.json 양쪽에서 feature 목록을 찾지 못했습니다."
        )

    return model, feature_cols


# ============================================================
# Evaluation helpers
# ============================================================
def validate_no_leak_features(feature_cols: list[str]) -> list[str]:
    leak_keywords = [
        "future_",
        "target",
        "target_date_",
        "answer_",
        "actual_",
        "predicted_",
        "error",
    ]

    return [
        col for col in feature_cols if any(keyword in col for keyword in leak_keywords)
    ]


def filter_test_period(
    df: pd.DataFrame,
    oil: str,
    model_type: str,
    train_cutoff: str | None,
) -> tuple[pd.DataFrame, str]:
    """
    평가/그래프 표시 기간 정의.

    default:
      2025-01-01 ~ 2025-12-31

    shock_aware:
      train_cutoff 이후 기간
      예: train_cutoff=2026-03-31이면 2026-04-01 이후
    """
    if model_type == "default":
        filtered = df[
            (df["date"] >= DEFAULT_TEST_START) & (df["date"] <= DEFAULT_TEST_END)
        ].copy()

        cutoff_mode = (
            f"default_test_period_"
            f"{DEFAULT_TEST_START.date()}_to_{DEFAULT_TEST_END.date()}"
        )

        if filtered.empty:
            raise ValueError(
                f"{oil} default 모델의 2025 테스트 기간 평가 데이터가 비어 있습니다. "
                f"기간={DEFAULT_TEST_START.date()}~{DEFAULT_TEST_END.date()}"
            )

        return filtered, cutoff_mode

    if model_type == "shock_aware":
        if not train_cutoff:
            raise ValueError(
                f"{oil} shock_aware 모델에 train_cutoff가 없습니다. "
                f"테스트 기간을 정의할 수 없습니다."
            )

        cutoff = pd.to_datetime(train_cutoff, errors="coerce")

        if pd.isna(cutoff):
            raise ValueError(f"{oil} train_cutoff 파싱 실패: {train_cutoff}")

        filtered = df[df["date"] > cutoff].copy()
        cutoff_mode = f"shock_aware_after_{cutoff.date()}"

        if filtered.empty:
            raise ValueError(
                f"{oil} shock_aware 테스트 기간 평가 데이터가 비어 있습니다. "
                f"train_cutoff={train_cutoff}"
            )

        return filtered, cutoff_mode

    raise ValueError(f"지원하지 않는 model_type입니다: {model_type}")


def attach_actual_future_price(
    df: pd.DataFrame,
    oil: str,
) -> pd.DataFrame:
    """
    target 값을 이용해 actual future price를 계산하지 않는다.

    정확한 평가 방식:
      현재 row의 target_date에 해당하는 실제 유가를 찾아서
      actual_price_10d로 사용한다.
    """
    current_col = CURRENT_PRICE_COLS[oil]

    if "target_date" not in df.columns:
        raise ValueError(
            f"{oil} dataset에 target_date 컬럼이 없습니다. "
            f"build_dataset.py에서 target_date를 생성해야 합니다."
        )

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["target_date"] = pd.to_datetime(base["target_date"], errors="coerce")
    base[current_col] = pd.to_numeric(base[current_col], errors="coerce")

    future_lookup = (
        base[["date", current_col]]
        .dropna(subset=["date", current_col])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .rename(
            columns={
                "date": "target_date",
                current_col: "actual_price_10d",
            }
        )
    )

    merged = base.merge(
        future_lookup,
        on="target_date",
        how="left",
    )

    merged["actual_price_10d"] = pd.to_numeric(
        merged["actual_price_10d"],
        errors="coerce",
    )

    merged["actual_return_decimal"] = (
        merged["actual_price_10d"] - merged[current_col]
    ) / merged[current_col]

    merged["actual_return_pct"] = merged["actual_return_decimal"] * 100.0

    return merged


def prepare_eval_data(
    df: pd.DataFrame,
    oil: str,
    model_type: str,
    feature_cols: list[str],
    train_cutoff: str | None,
):
    current_col = CURRENT_PRICE_COLS[oil]

    required = ["date", current_col, "target", "target_date"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{oil} dataset에 필요한 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    df[current_col] = pd.to_numeric(df[current_col], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    df = df.dropna(subset=["date", "target_date", current_col, "target"])
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{oil} target 유효 데이터가 비어 있습니다.")

    df = attach_actual_future_price(df, oil)

    before_actual_drop = len(df)
    df = df.dropna(
        subset=[
            "actual_price_10d",
            "actual_return_decimal",
            "actual_return_pct",
        ]
    ).copy()
    after_actual_drop = len(df)

    dropped_unknown_future = before_actual_drop - after_actual_drop

    if df.empty:
        raise ValueError(
            f"{oil} 실제 미래 가격을 찾을 수 있는 평가 데이터가 없습니다. "
            f"target_date가 현재 데이터셋의 date 범위 안에 있는지 확인하세요."
        )

    df, cutoff_mode = filter_test_period(
        df=df,
        oil=oil,
        model_type=model_type,
        train_cutoff=train_cutoff,
    )

    print(
        f"[INFO] {oil} test/eval range:",
        df["date"].min().date(),
        "~",
        df["date"].max().date(),
        "| rows:",
        len(df),
        "| cutoff_mode:",
        cutoff_mode,
    )

    if dropped_unknown_future > 0:
        print(
            f"[INFO] {oil} 실제 미래 가격을 아직 알 수 없어 평가에서 제외된 행:",
            dropped_unknown_future,
        )

    target_scale = "percent"

    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        print(f"[WARN] {oil} missing features in dataset: {len(missing_features)}")
        print("[WARN] missing sample:", missing_features[:30])

        for col in missing_features:
            df[col] = 0

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0)

    current_price = df[current_col].astype(float)
    actual_future_price = df["actual_price_10d"].astype(float)
    actual_return_decimal = df["actual_return_decimal"].astype(float)
    actual_return_pct = df["actual_return_pct"].astype(float)

    return (
        df,
        X,
        current_price,
        actual_future_price,
        actual_return_decimal,
        actual_return_pct,
        target_scale,
    )


def evaluate_one_model(
    oil: str,
    model_type: str,
    manifest: dict[str, Any],
    features_json: dict[str, Any],
) -> dict[str, Any]:
    dataset_path = DATASET_PATHS[oil]
    model_path = MODEL_PATHS[oil][model_type]
    model_file_name = model_path.name

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset이 없습니다: {dataset_path}")

    df = pd.read_csv(dataset_path)

    fallback_features = get_feature_list(features_json, oil, model_type)

    if not fallback_features:
        print(
            f"[WARN] final_model_features.json에서 feature 목록을 못 찾음: "
            f"{oil} / {model_type}"
        )

    model_obj = load_model(model_path)
    model, feature_cols = extract_model_and_features(model_obj, fallback_features)

    leak_features = validate_no_leak_features(feature_cols)

    if leak_features:
        raise ValueError(f"누수 feature 발견: {model_file_name}: {leak_features[:30]}")

    meta = get_model_meta(manifest, features_json, oil, model_type)

    model_name = str(meta.get("model_name", model_type))
    train_cutoff = meta.get("train_cutoff")
    feature_set = str(meta.get("feature_set", "unknown"))

    (
        eval_df,
        X,
        current_price,
        actual_future_price,
        actual_return_decimal,
        actual_return_pct,
        target_scale,
    ) = prepare_eval_data(
        df=df,
        oil=oil,
        model_type=model_type,
        feature_cols=feature_cols,
        train_cutoff=train_cutoff,
    )

    raw_pred = pd.Series(model.predict(X), index=X.index).astype(float)
    pred_return_pct = raw_pred
    pred_return_decimal = pred_return_pct / 100.0
    pred_future_price = current_price * (1 + pred_return_decimal)

    price_mae = mean_absolute_error(actual_future_price, pred_future_price)
    price_rmse = np.sqrt(mean_squared_error(actual_future_price, pred_future_price))
    price_r2 = r2_score(actual_future_price, pred_future_price)

    return_mae = mean_absolute_error(actual_return_pct, pred_return_pct)
    return_rmse = np.sqrt(mean_squared_error(actual_return_pct, pred_return_pct))
    return_r2 = r2_score(actual_return_pct, pred_return_pct)

    naive_future_price = current_price.copy()
    naive_price_mae = mean_absolute_error(actual_future_price, naive_future_price)
    naive_price_rmse = np.sqrt(
        mean_squared_error(actual_future_price, naive_future_price)
    )
    naive_price_r2 = r2_score(actual_future_price, naive_future_price)

    result_df = pd.DataFrame(
        {
            "date": eval_df["date"].values,
            "target_date": eval_df["target_date"].values,
            "current_price": current_price.values,
            "actual_price_10d": actual_future_price.values,
            "predicted_price_10d": pred_future_price.values,
            "naive_current_price": naive_future_price.values,
            "actual_return_decimal": actual_return_decimal.values,
            "predicted_return_decimal": pred_return_decimal.values,
            "actual_return_pct": actual_return_pct.values,
            "predicted_return_pct": pred_return_pct.values,
            "raw_model_prediction": raw_pred.values,
            "price_error": pred_future_price.values - actual_future_price.values,
            "return_error_pct": pred_return_pct.values - actual_return_pct.values,
        }
    )

    result_df = result_df.sort_values("target_date").reset_index(drop=True)

    return {
        "oil": oil,
        "model_type": model_type,
        "model_file": model_file_name,
        "model_name": model_name,
        "feature_set": feature_set,
        "train_cutoff": train_cutoff,
        "target_scale": target_scale,
        "feature_count": len(feature_cols),
        "eval_rows": len(result_df),
        "eval_start": str(result_df["date"].min().date()),
        "eval_end": str(result_df["date"].max().date()),
        "target_date_start": str(result_df["target_date"].min().date()),
        "target_date_end": str(result_df["target_date"].max().date()),
        "price_mae": price_mae,
        "price_rmse": price_rmse,
        "price_r2": price_r2,
        "return_mae": return_mae,
        "return_rmse": return_rmse,
        "return_r2": return_r2,
        "naive_price_mae": naive_price_mae,
        "naive_price_rmse": naive_price_rmse,
        "naive_price_r2": naive_price_r2,
        "result_df": result_df,
    }


# ============================================================
# Plot
# ============================================================
def plot_tracking_panel(results: list[dict[str, Any]], model_type: str):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(18, 12),
        sharex=True,
    )

    mode_label = f"{model_type.replace('_', ' ').title()} Mode"

    for ax, result in zip(axes, results):
        oil = result["oil"]
        oil_name = OIL_DISPLAY_NAMES[oil]
        model_name = result["model_name"].upper()
        df = result["result_df"].copy()

        ax.plot(
            df["target_date"],
            df["actual_price_10d"],
            linewidth=1.8,
            label="Actual 10D future price",
        )

        ax.plot(
            df["target_date"],
            df["predicted_price_10d"],
            linestyle="--",
            linewidth=1.6,
            label=f"AI predicted ({model_name})",
        )

        ax.plot(
            df["target_date"],
            df["naive_current_price"],
            linestyle=":",
            linewidth=1.2,
            label="Naive current price",
        )

        ax.set_title(
            f"{oil_name} [{mode_label}] - Test Period Tracking "
            f"({result['eval_start']} ~ {result['eval_end']}) | "
            f"AI MAE=${result['price_mae']:.2f}, "
            f"Naive MAE=${result['naive_price_mae']:.2f}, "
            f"AI R²={result['price_r2']:.3f}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        ax.set_ylabel("Price (USD / Barrel)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", frameon=False)

    axes[-1].set_xlabel("Date")

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"model_tracking_{model_type}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] plot saved -> {output_path}")


def plot_error_panel(results: list[dict[str, Any]], model_type: str):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(18, 10),
        sharex=True,
    )

    mode_label = f"{model_type.replace('_', ' ').title()} Mode"

    for ax, result in zip(axes, results):
        oil = result["oil"]
        oil_name = OIL_DISPLAY_NAMES[oil]
        df = result["result_df"].copy()

        ax.axhline(0, linewidth=1.0, alpha=0.8)
        ax.plot(
            df["target_date"],
            df["price_error"],
            linewidth=1.4,
            label="AI price error",
        )

        ax.set_title(
            f"{oil_name} [{mode_label}] - Test Period Error "
            f"({result['eval_start']} ~ {result['eval_end']})",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        ax.set_ylabel("USD")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", frameon=False)

    axes[-1].set_xlabel("Date")

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"model_error_{model_type}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] error plot saved -> {output_path}")


def plot_return_panel(results: list[dict[str, Any]], model_type: str):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(18, 10),
        sharex=True,
    )

    mode_label = f"{model_type.replace('_', ' ').title()} Mode"

    for ax, result in zip(axes, results):
        oil = result["oil"]
        oil_name = OIL_DISPLAY_NAMES[oil]
        df = result["result_df"].copy()

        ax.plot(
            df["target_date"],
            df["actual_price_10d"],
            linewidth=1.5,
            label="Actual 10D future price",
        )

        ax.plot(
            df["target_date"],
            df["predicted_price_10d"],
            linestyle="--",
            linewidth=1.4,
            label="Predicted 10D future price",
        )

        ax.set_title(
            f"{oil_name} [{mode_label}] - Price Tracking "
            f"({result['eval_start']} ~ {result['eval_end']}) | "
            f"Price MAE=${result['price_mae']:.2f}, "
            f"R²={result['price_r2']:.3f}",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        ax.set_ylabel("Price (USD / Barrel)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", frameon=False)

    axes[-1].set_xlabel("Date")

    fig.tight_layout()

    output_path = OUTPUT_DIR / f"model_return_{model_type}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] return plot saved -> {output_path}")


# ============================================================
# Save
# ============================================================
def save_result_csv(results: list[dict[str, Any]], model_type: str):
    metrics_rows = []

    for result in results:
        oil = result["oil"]

        metrics_rows.append(
            {
                "oil": oil,
                "model_type": model_type,
                "model_file": result["model_file"],
                "model_name": result["model_name"],
                "feature_set": result["feature_set"],
                "train_cutoff": result["train_cutoff"],
                "target_scale": result["target_scale"],
                "feature_count": result["feature_count"],
                "eval_rows": result["eval_rows"],
                "eval_start": result["eval_start"],
                "eval_end": result["eval_end"],
                "target_date_start": result["target_date_start"],
                "target_date_end": result["target_date_end"],
                "price_mae": result["price_mae"],
                "price_rmse": result["price_rmse"],
                "price_r2": result["price_r2"],
                "return_mae": result["return_mae"],
                "return_rmse": result["return_rmse"],
                "return_r2": result["return_r2"],
                "naive_price_mae": result["naive_price_mae"],
                "naive_price_rmse": result["naive_price_rmse"],
                "naive_price_r2": result["naive_price_r2"],
            }
        )

        pred_path = OUTPUT_DIR / f"{oil}_{model_type}_tracking_predictions.csv"
        result["result_df"].to_csv(pred_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] prediction csv saved -> {pred_path}")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = OUTPUT_DIR / f"model_tracking_metrics_{model_type}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] metrics csv saved -> {metrics_path}")


def save_combined_metrics(all_results: list[dict[str, Any]]):
    rows = []

    for result in all_results:
        rows.append(
            {
                "oil": result["oil"],
                "model_type": result["model_type"],
                "model_file": result["model_file"],
                "model_name": result["model_name"],
                "feature_set": result["feature_set"],
                "train_cutoff": result["train_cutoff"],
                "target_scale": result["target_scale"],
                "feature_count": result["feature_count"],
                "eval_rows": result["eval_rows"],
                "eval_start": result["eval_start"],
                "eval_end": result["eval_end"],
                "target_date_start": result["target_date_start"],
                "target_date_end": result["target_date_end"],
                "price_mae": result["price_mae"],
                "price_rmse": result["price_rmse"],
                "price_r2": result["price_r2"],
                "return_mae": result["return_mae"],
                "return_rmse": result["return_rmse"],
                "return_r2": result["return_r2"],
                "naive_price_mae": result["naive_price_mae"],
                "naive_price_rmse": result["naive_price_rmse"],
                "naive_price_r2": result["naive_price_r2"],
            }
        )

    df = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / "model_tracking_metrics_all.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] combined metrics saved -> {output_path}")


# ============================================================
# Runner
# ============================================================
def run_one_model_type(model_type: str) -> list[dict[str, Any]]:
    manifest = load_json(MANIFEST_PATH)
    features_json = load_json(FEATURES_PATH)

    results = []

    for oil in OIL_TYPES:
        print("=" * 80)
        print(f"Evaluating {oil.upper()} / {model_type}")
        print("=" * 80)

        result = evaluate_one_model(
            oil=oil,
            model_type=model_type,
            manifest=manifest,
            features_json=features_json,
        )

        print(
            f"PRICE: AI_MAE=${result['price_mae']:.4f}, "
            f"AI_RMSE=${result['price_rmse']:.4f}, "
            f"AI_R2={result['price_r2']:.4f}"
        )
        print(
            f"NAIVE: MAE=${result['naive_price_mae']:.4f}, "
            f"RMSE=${result['naive_price_rmse']:.4f}, "
            f"R2={result['naive_price_r2']:.4f}"
        )
        print(
            f"RETURN: MAE={result['return_mae']:.4f}p, "
            f"RMSE={result['return_rmse']:.4f}p, "
            f"R2={result['return_r2']:.4f}"
        )
        print(
            f"MODEL: {result['model_name']} | "
            f"feature_set={result['feature_set']} | "
            f"target_scale={result['target_scale']} | "
            f"features={result['feature_count']} | "
            f"eval_rows={result['eval_rows']} | "
            f"eval_period={result['eval_start']}~{result['eval_end']} | "
            f"target_date={result['target_date_start']}~{result['target_date_end']}"
        )

        results.append(result)

    plot_tracking_panel(results, model_type)
    plot_error_panel(results, model_type)
    plot_return_panel(results, model_type)
    save_result_csv(results, model_type)

    return results


def main():
    all_results = []

    for model_type in MODEL_TYPES:
        results = run_one_model_type(model_type)
        all_results.extend(results)

    save_combined_metrics(all_results)

    print("\n" + "=" * 80)
    print("모델 성능 추적 결과 생성 완료")
    print("=" * 80)
    print("output dir:", OUTPUT_DIR)
    print("주요 확인 파일:")
    print("-", OUTPUT_DIR / "model_tracking_default.png")
    print("-", OUTPUT_DIR / "model_tracking_shock_aware.png")
    print("-", OUTPUT_DIR / "model_error_default.png")
    print("-", OUTPUT_DIR / "model_error_shock_aware.png")
    print("-", OUTPUT_DIR / "model_return_default.png")
    print("-", OUTPUT_DIR / "model_return_shock_aware.png")
    print("-", OUTPUT_DIR / "model_tracking_metrics_all.csv")


if __name__ == "__main__":
    main()
