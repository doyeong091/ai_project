# model/train.py
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

from config import (
    CURRENT_PRICE_COLUMNS, DATASET_PATHS, EXPERIMENT_DIR, PRIMARY_OIL_TYPE,
    SELECTED_FEATURES_PATH, TARGET_COL, TRAIN_RESULTS_JSON_PATH, 
    TRAIN_RESULTS_PATH, TRAIN_RATIO, VAL_RATIO, ensure_directories,
)

def load_selected_features() -> dict:
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_feature_list(df: pd.DataFrame, features: list[str]) -> list[str]:
    numeric_cols = set(df.select_dtypes(include=["number"]).columns)
    leak_keywords = ["future_", "target", "target_date", "answer_", "actual_", "predicted_", "error"]
    return [c for c in features if c in numeric_cols and c != "date" and not any(k in c for k in leak_keywords)]

# [핵심 수정 1] 훈련 데이터의 불균형 비율에 맞춰 가중치(pos_weight)를 동적으로 받도록 변경
def make_hybrid_models(pos_weight: float) -> dict:
    return {
        "hybrid_rf_ridge": {
            "classifier": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBClassifier(n_estimators=300, max_depth=4, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1))
            ]),
            "normal_reg": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1))
            ]),
            "shock_reg": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0))
            ])
        }
    }

def time_series_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

def make_xy(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    y = df[TARGET_COL].copy()
    y_shock = df['target_shock'].copy() if 'target_shock' in df.columns else np.zeros(len(df))
    return X, y, y_shock

def evaluate_prediction(df: pd.DataFrame, pred_change: np.ndarray, current_col: str) -> dict:
    y_true = df[TARGET_COL].to_numpy() 
    curr = df[current_col].to_numpy()
    
    n_mask = np.abs(y_true) < 15
    s15_mask = np.abs(y_true) >= 15
    
    def calc_mae(mask): 
        return float(mean_absolute_error(y_true[mask], pred_change[mask])) if mask.sum() > 0 else float("nan")
    
    true_price = curr * (1 + y_true / 100)
    pred_price = curr * (1 + pred_change / 100)
    price_mae = float(mean_absolute_error(true_price, pred_price))

    return {
        "target_mae_pct": float(mean_absolute_error(y_true, pred_change)),
        "normal_mae_pct": calc_mae(n_mask),
        "shock_gt_15_mae_pct": calc_mae(s15_mask),
        "price_mae_dollar": price_mae
    }

def main():
    ensure_directories()
    oil_type = PRIMARY_OIL_TYPE
    df = pd.read_csv(DATASET_PATHS[oil_type]).sort_values("date").reset_index(drop=True)
    train_df, val_df, test_df = time_series_split(df)
    feature_sets = load_selected_features()["feature_sets"]
    
    results = []

    for f_name, raw_cols in feature_sets.items():
        f_cols = clean_feature_list(df, raw_cols)
        X_train, y_train, y_train_shock = make_xy(train_df, f_cols)
        X_test, y_test, y_test_shock = make_xy(test_df, f_cols)
        
        # [핵심 수정 2] 동적 가중치 계산 (정상 장세 수 / 쇼크 장세 수)
        # 만약 쇼크가 0이면 에러가 나므로 max(1, ...) 처리
        dynamic_pos_weight = (y_train_shock == 0).sum() / max(1, (y_train_shock == 1).sum())
        hybrid_dict = make_hybrid_models(pos_weight=dynamic_pos_weight)

        for model_name, parts in hybrid_dict.items():
            clf, norm_reg, shock_reg = parts["classifier"], parts["normal_reg"], parts["shock_reg"]
            
            # 1. 분류기 학습
            clf.fit(X_train, y_train_shock)
            
            # 2. 회귀 모델 학습 (정상/쇼크 분할)
            norm_mask = (y_train_shock == 0)
            shock_mask = (y_train_shock == 1)
            
            norm_reg.fit(X_train[norm_mask], y_train[norm_mask])
            if shock_mask.sum() > 0:
                shock_reg.fit(X_train[shock_mask], y_train[shock_mask])
            else:
                shock_reg = norm_reg

            # 3. Test 예측
            # [핵심 수정 3] 경보 문턱값(Threshold)을 0.50에서 0.20으로 대폭 하향하여 민감도 증폭
            CUSTOM_THRESHOLD = 0.10
            shock_probs = clf.predict_proba(X_test)[:, 1]
            pred_shock_labels = (shock_probs >= CUSTOM_THRESHOLD).astype(int) 
            
            test_pred = np.zeros(len(X_test))
            
            mask_norm = (pred_shock_labels == 0)
            mask_shock = (pred_shock_labels == 1)
            
            if mask_norm.any():
                test_pred[mask_norm] = norm_reg.predict(X_test[mask_norm])
            if mask_shock.any():
                test_pred[mask_shock] = shock_reg.predict(X_test[mask_shock])
                
            # 평가 지표 산출
            metrics = evaluate_prediction(test_df, test_pred, CURRENT_PRICE_COLUMNS[oil_type])
            
            row = {"feature_set": f_name, "model": model_name, **metrics}
            results.append(row)
            
            actual_shocks = int(y_test_shock.sum())
            total_alarms = int(pred_shock_labels.sum())
            true_hits = int(((y_test_shock == 1) & (pred_shock_labels == 1)).sum())
            
            print(f"[{f_name}] {model_name:15s}")
            print(f"   -> 🔍 팩트체크: 실제 쇼크 {actual_shocks}일 중 {true_hits}일 적중 (총 경보 횟수: {total_alarms}회)")
            print(f"   -> 📈 오차분석: 등락률 MAE={metrics['target_mae_pct']:.2f}%p (쇼크 시 {metrics['shock_gt_15_mae_pct']:.2f}%p)")
            print(f"   -> 💵 가격오차: 예측 달러 MAE=${metrics['price_mae_dollar']:.2f}")
            print("-" * 60)

    pd.DataFrame(results).to_csv(TRAIN_RESULTS_PATH, index=False, encoding="utf-8-sig")
    with open(TRAIN_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("하이브리드 파이프라인 학습 완료 및 저장 성공!")

if __name__ == "__main__":
    main()