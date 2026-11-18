import os
import re
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import TimeSeriesSplit

from feature_engineering import generate_features

print("🚀 勝ち子ちゃん 【地方特化・特徴量精鋭化・完全同期アンサンブル v2.1】")

RAW_CSV_FILE = "nar_keiba_database.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"

if not os.path.exists(RAW_CSV_FILE):
    print(f"⚠️ 生データ '{RAW_CSV_FILE}' が見つかりません。")
    exit()

print(f"📊 生データ '{RAW_CSV_FILE}' を読み込み中...")
df_raw = pd.read_csv(RAW_CSV_FILE, low_memory=False, encoding='utf-8-sig')

# 1. 厳格な時系列ソート
if 'date' in df_raw.columns:
    df_raw['date_parsed'] = pd.to_datetime(df_raw['date'], errors='coerce')
    df_raw = df_raw.sort_values(['date_parsed', 'race_id']).reset_index(drop=True)
else:
    df_raw = df_raw.sort_values('race_id').reset_index(drop=True)

# 2. 目的変数（relevance）の定義
clean_rank = pd.to_numeric(df_raw['着順'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df_raw['relevance'] = clean_rank.map({1: 5, 2: 3, 3: 2, 4: 1, 5: 1}).fillna(0).astype(int)
df_raw['target_win'] = (clean_rank == 1).astype(float)
df_raw['target_place'] = (clean_rank <= 3).astype(float)

# 有効レースの抽出（着順データがあるレースのみ）
valid_rids = df_raw.groupby('race_id')['relevance'].max()
valid_rids = valid_rids[valid_rids > 0].index
df_raw = df_raw[df_raw['race_id'].isin(valid_rids)].reset_index(drop=True)

race_groups = df_raw.groupby('race_id', sort=False).size().values
unique_races = df_raw['race_id'].unique()

print(f"✨ 全 {len(df_raw):,} 件 / {len(unique_races):,} レース を認識")

# ==========================================
# 🚨 ホワイトリスト：gate_num（馬番）を完全排除
# ==========================================
safe_base_cols = [
    # 基本条件（gate_num は排除、waku_num は維持）
    'distance_num', 'is_turf', 'turn_direction', 'is_heavy_track',
    'waku_num', 'weight_num', 'body_weight', 'weight_change',
    'kinryo_ratio', 'weight_change_ratio', 'kinryo_diff',
    # 脚質・先行力
    'is_style_nige', 'is_style_sen', 'is_style_sashi', 'is_style_oikomi',
    'start_dash_index', 'early_pace_index', 'is_jockey_changed',
    # キャリア・出走数
    'valid_runs_count', 'is_no_past_data',
    # 集計指数
    'horse_mean_speed', 'horse_max_speed', 'horse_min_speed', 'horse_std_speed',
    'horse_mean_last3f', 'horse_min_last3f', 'horse_max_last3f', 'horse_mean_1c',
    'ema3_custom_time_index_m', 'ema5_custom_time_index_m', 'best_custom_time_index_m',
    'ema3_custom_last3f_index', 'ema3_custom_start_index',
    # トレンド
    'speed_trend_1_2', 'speed_trend_1_3', 'speed_trend_ema', 'last3f_trend_1_2', 'pos_trend_1_2',
    # 騎手・厩舎実績
    'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'jockey_course_win_rate',
    # 相互作用
    'interaction_time_jockey_win', 'interaction_time_jockey_place',
    'interaction_time_course_jockey', 'interaction_best_jockey',
    'interaction_ema5_trainer', 'interaction_start_last3f',
    'interaction_front_heavy', 'interaction_style_waku',
    'interaction_weight_fatigue', 'interaction_weight_speed',
    'interaction_trainer_jockey', 'interaction_waku_jockey'
]

# 前走〜前5走の実績データ
for i in range(1, 6):
    safe_base_cols.extend([
        f'prev{i}_speed_idx_safe',
        f'prev{i}_last3f_safe',
        f'prev{i}_1c_safe',
        f'prev{i}_4c_safe',
        f'prev{i}_time_diff_safe',
        f'prev{i}_pop_safe',
        f'prev{i}_odds_safe'
    ])

# 相対特徴量を作成する対象
rel_targets = [
    'body_weight', 'weight_change', 'kinryo_ratio', 'weight_change_ratio',
    'start_dash_index', 'early_pace_index',
    'horse_mean_speed', 'horse_max_speed', 'horse_min_speed', 'horse_std_speed',
    'horse_mean_last3f', 'horse_min_last3f', 'horse_max_last3f', 'horse_mean_1c',
    'speed_trend_1_2', 'speed_trend_1_3', 'speed_trend_ema', 'last3f_trend_1_2', 'pos_trend_1_2',
    'ema3_custom_time_index_m', 'ema5_custom_time_index_m', 'best_custom_time_index_m',
    'ema3_custom_last3f_index', 'ema3_custom_start_index',
    'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'jockey_course_win_rate',
    'interaction_time_jockey_win', 'interaction_time_jockey_place',
    'interaction_time_course_jockey', 'interaction_best_jockey',
    'interaction_ema5_trainer', 'interaction_start_last3f',
    'interaction_front_heavy', 'interaction_style_waku',
    'interaction_weight_fatigue', 'interaction_weight_speed',
    'interaction_trainer_jockey', 'interaction_waku_jockey'
]
for i in range(1, 6):
    rel_targets.extend([
        f'prev{i}_speed_idx_safe',
        f'prev{i}_last3f_safe',
        f'prev{i}_1c_safe',
        f'prev{i}_time_diff_safe'
    ])

safe_rel_cols = []
for col in rel_targets:
    safe_rel_cols.extend([
        f'{col}_race_zscore',
        f'{col}_race_rank'
    ])

allowed_features = set(safe_base_cols + safe_rel_cols)

# ==========================================
# 【Optuna】 Fold内動的生成 目的関数
# ==========================================
def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': trial.suggest_int('n_estimators', 150, 350),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.07),
        'num_leaves': trial.suggest_int('num_leaves', 24, 96),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 80),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'subsample_freq': 1,
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []

    for fold, (train_g_idx, valid_g_idx) in enumerate(tscv.split(race_groups)):
        train_races = unique_races[train_g_idx]
        valid_races = unique_races[valid_g_idx]

        raw_train = df_raw[df_raw['race_id'].isin(train_races)].copy()
        raw_valid = df_raw[df_raw['race_id'].isin(valid_races)].copy()

        feat_train = generate_features(raw_train, is_train=True)
        sort_cols_tr = ['date_parsed', 'race_id'] if 'date_parsed' in feat_train.columns else ['race_id']
        feat_train = feat_train.sort_values(sort_cols_tr).reset_index(drop=True)

        past_j = feat_train.groupby('騎手_clean')[['jockey_win_rate', 'jockey_place_rate']].last().reset_index()
        past_t = feat_train.groupby('調教師_clean')[['trainer_win_rate']].last().reset_index()
        past_jp = feat_train.groupby('j_place_key')[['jockey_course_win_rate']].last().reset_index()

        feat_valid = generate_features(
            raw_valid, is_train=False,
            past_j_stats=past_j, past_t_stats=past_t, past_jp_stats=past_jp
        )
        sort_cols_va = ['date_parsed', 'race_id'] if 'date_parsed' in feat_valid.columns else ['race_id']
        feat_valid = feat_valid.sort_values(sort_cols_va).reset_index(drop=True)

        f_cols = [c for c in feat_train.columns if c in allowed_features and pd.api.types.is_numeric_dtype(feat_train[c])]

        X_tr = feat_train[f_cols].astype(float)
        y_tr = feat_train['relevance']
        g_tr = feat_train.groupby('race_id', sort=False).size().values

        X_va = feat_valid[f_cols].astype(float)
        y_va = feat_valid['relevance']
        g_va = feat_valid.groupby('race_id', sort=False).size().values

        model = lgb.LGBMRanker(**params)
        try:
            model.fit(
                X_tr, y_tr, group=g_tr,
                eval_set=[(X_va, y_va)], eval_group=[g_va],
                eval_at=[5], callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            score = model.best_score_['valid_0']['ndcg@5']
            fold_scores.append(score)
        except Exception as e:
            print(f"⚠️ Fold {fold} スキップ: {e}")
            return 0.0

    return np.mean(fold_scores) if fold_scores else 0.0

print("\n--- 🔍 Optuna ハイパーパラメータ最適化 (精鋭特徴量・正則化強化) ---")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

best_params = study.best_params
print(f"\n🏆 最高スコア (真のNDCG@5): {study.best_value:.4f}")
print(f"⚙️ 最適パラメータ: {best_params}")
best_params.update({
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'subsample_freq': 1,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
})

# ==========================================
# 【本番学習】 全データに対して最終モデルを構築
# ==========================================
print("\n--- 🏗️ 全データによる最終特徴量生成中... ---")
df_final_feat = generate_features(df_raw, is_train=True)
sort_cols_full = ['date_parsed', 'race_id'] if 'date_parsed' in df_final_feat.columns else ['race_id']
df_final_feat = df_final_feat.sort_values(sort_cols_full).reset_index(drop=True)

final_f_cols = [c for c in df_final_feat.columns if c in allowed_features and pd.api.types.is_numeric_dtype(df_final_feat[c])]

X_full = df_final_feat[final_f_cols].astype(float)
final_features = list(X_full.columns)

y_full = df_final_feat['relevance']
groups_full = df_final_feat.groupby('race_id', sort=False).size().values
cat_group_ids = df_final_feat.groupby('race_id', sort=False).ngroup()

print(f"✨ 本番学習データ総数: {len(X_full):,} 件 / {len(groups_full):,} レース")
print(f"🔥 最終投入特徴量数: {len(final_features)} 個")

print("\n--- 【1/3】 LightGBM Ranker 本番学習中... ---")
ranker_lgb = lgb.LGBMRanker(**best_params)
ranker_lgb.fit(X_full, y_full, group=groups_full)

imp_split = pd.Series(ranker_lgb.feature_importances_, index=final_features).sort_values(ascending=False)
imp_gain = pd.Series(ranker_lgb.booster_.feature_importance(importance_type='gain'), index=final_features).sort_values(ascending=False)

print("\n" + "!" * 50)
print("🚨 【実測：特徴量重要度 トップ15（実質貢献度 Gain）】")
print(imp_gain.head(15).round(1))
print("-" * 50)
print("📊 【参考：分岐回数 Split トップ10】")
print(imp_split.head(10))
print("!" * 50 + "\n")

print("\n--- 【2/3】 XGBoost Ranker 本番学習中... ---")
max_depth = max(3, min(7, int(np.log2(best_params.get('num_leaves', 31)))))
ranker_xgb = xgb.XGBRanker(
    objective='rank:ndcg',
    n_estimators=best_params.get('n_estimators', 200),
    learning_rate=best_params.get('learning_rate', 0.04),
    max_depth=max_depth,
    colsample_bytree=best_params.get('colsample_bytree', 0.7),
    subsample=best_params.get('subsample', 0.85),
    reg_alpha=best_params.get('reg_alpha', 1.0),
    reg_lambda=best_params.get('reg_lambda', 1.0),
    random_state=42,
    n_jobs=-1
)
ranker_xgb.fit(X_full, y_full, group=groups_full)

print("\n--- 【3/3】 CatBoost Ranker 本番学習中... ---")
ranker_cat = cb.CatBoostRanker(
    loss_function='YetiRank',
    iterations=best_params.get('n_estimators', 200),
    learning_rate=best_params.get('learning_rate', 0.04),
    depth=min(7, max_depth),
    l2_leaf_reg=max(1.0, best_params.get('reg_lambda', 3.0)),
    random_state=42,
    verbose=False,
    thread_count=-1
)
ranker_cat.fit(X_full, y_full, group_id=cat_group_ids)

# ==========================================
# 【最終保存】 最新の騎手・調教師マスタ同梱
# ==========================================
final_past_j = df_final_feat.groupby('騎手_clean')[['jockey_win_rate', 'jockey_place_rate']].last().reset_index()
final_past_t = df_final_feat.groupby('調教師_clean')[['trainer_win_rate']].last().reset_index()
final_past_jp = df_final_feat.groupby('j_place_key')[['jockey_course_win_rate']].last().reset_index()

joblib.dump({
    'model_rank_lgb': ranker_lgb,
    'model_rank_xgb': ranker_xgb,
    'model_rank_cat': ranker_cat,
    'features': final_features,
    'past_j_stats': final_past_j,
    'past_t_stats': final_past_t,
    'past_jp_stats': final_past_jp
}, MODEL_FILE)

print("=" * 50)
print(f"🎉 地方特化アンサンブルモデル保存完了: {MODEL_FILE}")
print("=" * 50 + "\n")