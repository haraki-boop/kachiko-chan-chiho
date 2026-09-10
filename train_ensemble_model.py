import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import os
import re
import optuna
from sklearn.model_selection import TimeSeriesSplit

print("🚀 勝ち子ちゃん 【LambdaMART 3連系特化版：完全リーク対策＆カンマエラー修正＆高速枝刈り】")

# 🌟 有効なデータ件数が存在するV2ファイルを自動読み込み
candidate_files = [
    "ml_target_data_v2.csv",
    "ml_target_data_chiho_v2.csv",
    "ml_target_data_chiho.csv",
    "ml_target_data.csv"
]

CSV_FILE = None
df = None

for fname in candidate_files:
    if os.path.exists(fname) and os.path.getsize(fname) > 100:
        try:
            try:
                temp_df = pd.read_csv(fname, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                temp_df = pd.read_csv(fname, low_memory=False, encoding='cp932')
            
            if len(temp_df) > 0:
                CSV_FILE = fname
                df = temp_df
                break
        except Exception:
            continue

if CSV_FILE is None or df is None or len(df) == 0:
    print("⚠️ 有効なデータファイルが見つかりません。")
    exit()

print(f"📊 データファイル '{CSV_FILE}' ({len(df)} 件) を読み込みました。")
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

target_cols_order = ['rank_num', '着順', '着順_num', 'target_rank_clean']
target_col = None
for col in target_cols_order:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("⚠️ 着順を示すカラムが見つかりません。")
    exit()

print(f"🎯 ターゲット着順カラムとして '{target_col}' を使用します。")
df['target_rank_clean'] = pd.to_numeric(df[target_col].apply(parse_rank), errors='coerce')
df = df[df['target_rank_clean'].notna() & (df['target_rank_clean'] < 90.0)].copy()

def rank_to_relevance(rank):
    if rank == 1.0: return 3
    elif rank == 2.0: return 2
    elif rank == 3.0: return 1
    else: return 0

df['relevance'] = df['target_rank_clean'].apply(rank_to_relevance)

date_col = 'date_parsed' if 'date_parsed' in df.columns else ('date' if 'date' in df.columns else None)
if date_col:
    df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(['date_parsed', 'race_id']).reset_index(drop=True)

df = df.copy()

# 🚨【超重要】カンニング（データリーク）を完全に防ぐための除外リスト（賞金・当日指数もブロック）
exclude_cols = set([
    'race_id', 'date', 'date_parsed', '馬名', '馬名_clean', '騎手', 'jockey_clean', 
    '調教師', 'trainer_clean', 'surface', 'condition', 'dist_cat', 'place_code_str', 
    'place_name', 'is_heavy_track', 'turn_direction', 'race_name', 'meet_day_num',
    'category', 'track_condition', 'place_code', '備考', '厩舎ｺﾒﾝﾄ', '調教ﾀｲﾑ',
    
    '着順', '着順_num', 'rank_num', 'target_rank_clean', 'relevance', 
    'is_win', 'is_rentai', 'target_win', 'target_place', 'target_rank',
    
    'time', 'タイム', 'time_seconds', 'time_sec_clean', 'time_sec', 'time_diff', '着差',
    'last_3f', 'last_3f_val', 'last3f_sec_clean', '上り', '上がり', '上がり3F',
    '通過', 'コーナー通過順', 'first_pos_clean', 'last_pos_clean', 'first_corner', 'last_corner', 'corner_diff',
    
    'race_avg_time', 'race_std_time', 'race_avg_last3f', 'race_std_last3f', 
    'race_avg_pos', 'race_std_pos', 'my_time_idx', 'my_last3f_idx', 'my_start_idx',
    'custom_time_index', 'custom_time_index_m', 'custom_start_index', 'custom_pursuit_index', 'custom_last3f_index',
    'mid_speed', 'hybrid_power_idx', 'my_pace_idx', 'pace_scenario_idx',
    
    '賞金(万円)', 'prize', 'prize_num',
    '単勝', '人気', 'オッズ', 'odds'
])

features = []
potential_cols = [c for c in df.columns if c not in exclude_cols]

# 🌟 カンマ(,)を取り除いてから安全に数値化する処理
for col in potential_cols:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    
    converted = pd.to_numeric(df[col], errors='coerce')
    if converted.notna().sum() > 0:
        df[col] = converted.fillna(0.0)
        features.append(col)

X = df[features].astype(float)
y_relevance = df['relevance']

groups = df.groupby('race_id', sort=False).size().values

print(f"✨ 厳選・自動検出された {len(features)} 個の合法な特徴量（前走実績など）を使用します。")
print(f"✨ 全 {len(df)} 件 / {len(groups)} レースのグループ構造で学習します...")

if len(df) == 0 or len(groups) < 4:
    print(f"⚠️ 学習可能なデータ件数/レース数が不足しています。(件数: {len(df)}, レース数: {len(groups)})")
    exit()

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 64),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 300),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []
    
    for fold, (train_g_idx, valid_g_idx) in enumerate(tscv.split(groups)):
        train_groups_fold = groups[train_g_idx]
        valid_groups_fold = groups[valid_g_idx]
        
        train_rows = int(np.sum(train_groups_fold))
        valid_rows = int(np.sum(valid_groups_fold))
        
        X_train_fold = X.iloc[:train_rows]
        y_train_fold = y_relevance.iloc[:train_rows]
        
        X_valid_fold = X.iloc[train_rows:train_rows + valid_rows]
        y_valid_fold = y_relevance.iloc[train_rows:train_rows + valid_rows]
        
        model = lgb.LGBMRanker(**params)
        try:
            model.fit(
                X_train_fold, y_train_fold, group=train_groups_fold,
                eval_set=[(X_valid_fold, y_valid_fold)], eval_group=[valid_groups_fold],
                eval_at=[3], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
            )
            score = model.best_score_['valid_0']['ndcg@3']
            fold_scores.append(score)
            
            # 🌟 Optuna枝刈り（Pruning）
            current_mean = np.mean(fold_scores)
            trial.report(current_mean, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        except optuna.TrialPruned:
            raise
        except Exception:
            return 0.0
            
    return np.mean(fold_scores) if fold_scores else 0.0

# 🌟 MedianPrunerを有効化
print("\n--- 🔍 Optuna 時系列CV ＆ 枝刈り（Pruning）実行中 (100試行)... ---")
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
study = optuna.create_study(direction='maximize', pruner=pruner)
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_params['objective'] = 'lambdarank'
best_params['metric'] = 'ndcg'
best_params['random_state'] = 42
best_params['verbose'] = -1

print(f"\n✨ 最適パラメータ発見: {best_params}")
print(f"🏆 最高スコア (NDCG@3): {study.best_value:.4f}")

print("\n--- 【1/3】 LightGBM Ranker 本番学習中... ---")
ranker_lgb = lgb.LGBMRanker(**best_params)
ranker_lgb.fit(X, y_relevance, group=groups)

print("\n--- 【2/3】 XGBoost Ranker 学習中... ---")
max_depth = max(3, int(np.log2(best_params.get('num_leaves', 32))))
ranker_xgb = xgb.XGBRanker(
    objective='rank:ndcg',
    n_estimators=best_params.get('n_estimators', 200),
    learning_rate=best_params.get('learning_rate', 0.05),
    max_depth=max_depth,
    random_state=42
)
ranker_xgb.fit(X, y_relevance, group=groups)

print("\n--- 【3/3】 CatBoost Ranker 学習中... ---")
ranker_cat = cb.CatBoostRanker(
    loss_function='YetiRank',
    iterations=best_params.get('n_estimators', 200),
    learning_rate=best_params.get('learning_rate', 0.05),
    depth=min(10, max_depth),
    random_state=42,
    verbose=False
)
ranker_cat.fit(X, y_relevance, group_id=df['race_id'])

joblib.dump({
    'model_rank_lgb': ranker_lgb,
    'model_rank_xgb': ranker_xgb,
    'model_rank_cat': ranker_cat,
    'features': features
}, MODEL_FILE)

importances = pd.Series(ranker_lgb.feature_importances_, index=features).sort_values(ascending=False)
print("\n" + "="*50)
print("📊 特徴量重要度 (Feature Importance Top 10):")
print(importances.head(10))
print("-" * 50)
print("⚠️ 効き目の薄い特徴量 (Bottom 5):")
print(importances.tail(5))
print("="*50)

print(f"\n✨ 反映完了: 3連系特化Rankingモデル（{MODEL_FILE}）の出力が完了しました！")