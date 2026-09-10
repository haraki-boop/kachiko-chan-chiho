import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import TimeSeriesSplit

print("🔍 最適パラメータでの NDCG@3 スコアを計算中...")

# データの読み込み
df = pd.read_csv("ml_target_data_chiho_v2.csv", low_memory=False)

# ターゲット（着順）の設定
df['target_rank_clean'] = pd.to_numeric(df['rank_num'], errors='coerce')
df = df[df['target_rank_clean'].notna() & (df['target_rank_clean'] < 90.0)].copy()

def rank_to_relevance(rank):
    if rank == 1.0: return 3
    elif rank == 2.0: return 2
    elif rank == 3.0: return 1
    else: return 0
df['relevance'] = df['target_rank_clean'].apply(rank_to_relevance)

# 日付ソート
df['date_parsed'] = pd.to_datetime(df['date_parsed'], errors='coerce')
df = df.sort_values(['date_parsed', 'race_id']).reset_index(drop=True)

# モデルと特徴量の読み込み
models = joblib.load("keiba_ai_model_nar_ensemble.pkl")
features = models['features']

# 🌟 カンマ(,)を取り除いてから数値化する安全な処理
for col in features:
    if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

X = df[features].astype(float)
y_relevance = df['relevance']
groups = df.groupby('race_id', sort=False).size().values

# 先ほど見つけた最適パラメータ
best_params = {
    'n_estimators': 248, 'learning_rate': 0.0786123985323588, 
    'num_leaves': 35, 'min_data_in_leaf': 233, 
    'colsample_bytree': 0.6573013559687304, 'subsample': 0.8804872776891353, 
    'objective': 'lambdarank', 'metric': 'ndcg', 'random_state': 42, 'verbose': -1
}

tscv = TimeSeriesSplit(n_splits=3)
fold_scores = []

for train_g_idx, valid_g_idx in tscv.split(groups):
    train_groups_fold = groups[train_g_idx]
    valid_groups_fold = groups[valid_g_idx]
    train_rows = int(np.sum(train_groups_fold))
    valid_rows = int(np.sum(valid_groups_fold))
    
    X_train_fold, y_train_fold = X.iloc[:train_rows], y_relevance.iloc[:train_rows]
    X_valid_fold, y_valid_fold = X.iloc[train_rows:train_rows+valid_rows], y_relevance.iloc[train_rows:train_rows+valid_rows]
    
    model = lgb.LGBMRanker(**best_params)
    model.fit(
        X_train_fold, y_train_fold, group=train_groups_fold,
        eval_set=[(X_valid_fold, y_valid_fold)], eval_group=[valid_groups_fold],
        eval_at=[3], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    fold_scores.append(model.best_score_['valid_0']['ndcg@3'])

print("\n" + "="*40)
print(f"🏆 カンニングなし・時系列CVの真のスコア")
print(f"NDCG@3 平均スコア: {np.mean(fold_scores):.4f}")
print("="*40)