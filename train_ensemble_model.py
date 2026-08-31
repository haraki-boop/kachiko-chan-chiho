import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import os
import re
import optuna

print("🚀 勝ち子ちゃん 【LambdaMART 3連系特化版：1着=3, 2着=2, 3着=1, 4着以下=0】")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み、前処理および特徴量を生成中...")
try:
    df = pd.read_csv(CSV_FILE, low_memory=False, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(CSV_FILE, low_memory=False, encoding='cp932')

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

target_col = '着順_num' if '着順_num' in df.columns else '着順'
df['target_rank_clean'] = df[target_col].apply(parse_rank)
df = df[df['target_rank_clean'].notna() & (df['target_rank_clean'] < 90.0)].copy()

def rank_to_relevance(rank):
    if rank == 1.0: return 3
    elif rank == 2.0: return 2
    elif rank == 3.0: return 1
    else: return 0

df['relevance'] = df['target_rank_clean'].apply(rank_to_relevance)

df['first_corner_raw'] = pd.to_numeric(df.get('first_corner', df.get('1角')), errors='coerce').fillna(8.0)
df['last_corner_raw'] = pd.to_numeric(df.get('last_corner', df.get('4角')), errors='coerce').fillna(df['first_corner_raw'])
df['corner_diff_raw'] = df['first_corner_raw'] - df['last_corner_raw']

df['last_3f'] = pd.to_numeric(df.get('last_3f', df.get('上り')), errors='coerce')
df['time_diff'] = pd.to_numeric(df.get('time_diff', df.get('着差')), errors='coerce').fillna(1.5)
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(0)
df['waku_num'] = pd.to_numeric(df.get('枠番'), errors='coerce').fillna(0)
df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce').fillna(1400)
df['馬名_clean'] = df['馬名'].astype(str).apply(lambda x: re.sub(r'[\s\u3000]+', '', str(x)))
df['date'] = pd.to_datetime(df.get('date', pd.Series(['2020-01-01']*len(df))), errors='coerce')

df['prize_num'] = pd.to_numeric(df.get('賞金(万円)', 0), errors='coerce').fillna(0.0)
df['prize_num_log'] = np.log1p(df['prize_num'])
df['horse_prize_avg'] = df.groupby('馬名_clean')['prize_num_log'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean().fillna(0.0))

df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

df['race_prize_mean'] = df.groupby('race_id')['horse_prize_avg'].transform('mean').clip(lower=0.1)
df['race_prize_relative'] = df['horse_prize_avg'] / df['race_prize_mean']
df['race_prize_rank'] = df.groupby('race_id')['horse_prize_avg'].rank(ascending=False, method='min')

MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df['place_code'] = df['race_id'].astype(str).str[4:6]
df['is_minami_kanto'] = df['place_code'].isin(MINAMI_KANTO_CODES).astype(int)

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0) if m else (470.0, 0.0)

if '馬体重' in df.columns:
    parsed = df['馬体重'].apply(parse_weight_info)
    df['body_weight'] = parsed.apply(lambda x: x[0])
    df['body_weight_diff'] = parsed.apply(lambda x: x[1])
else: 
    df['body_weight'] = 470.0
    df['body_weight_diff'] = 0.0

df['kinryo_weight_ratio'] = df['斤量'] / df['body_weight'].clip(lower=350.0)
df['is_large_weight_change'] = (df['body_weight_diff'].abs() >= 10.0).astype(int)

baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df['baba_code'] = df.get('馬場', pd.Series(['良']*len(df))).map(baba_map).fillna(1)
df['is_bad_baba'] = (df['baba_code'] >= 3).astype(int)

# 🌟 新規追加1：バテやすさ（失速率）フラグ
# 1角から4角にかけてポジションを3つ以上落としていたら失速と判定[cite: 4]
df['is_stalled'] = (df['last_corner_raw'] - df['first_corner_raw'] >= 3).astype(int)
df['prev_stall_rate'] = df.groupby('馬名_clean')['is_stalled'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean().fillna(0.0))

df['class_weighted_score'] = np.where(
    df['target_rank_clean'] <= 3.0, 
    (4.0 - df['target_rank_clean']) * df['race_prize_mean'], 
    0.0
)
df['prev_class_weighted_score'] = df.groupby('馬名_clean')['class_weighted_score'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(0.0))

df['custom_time_index'] = pd.to_numeric(df.get('custom_time_index'), errors='coerce').fillna(100.0)
df['custom_start_index'] = pd.to_numeric(df.get('custom_start_index'), errors='coerce').fillna(50.0)
df['custom_last3f_index'] = pd.to_numeric(df.get('custom_last3f_index'), errors='coerce').fillna(50.0)
df['dist_change_num'] = pd.to_numeric(df.get('dist_change'), errors='coerce').fillna(0.0)

df['prev_time_index_avg'] = df.groupby('馬名_clean')['custom_time_index'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(100.0))
df['prev_start_index_avg'] = df.groupby('馬名_clean')['custom_start_index'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(50.0))
df['prev_last3f_index_avg'] = df.groupby('馬名_clean')['custom_last3f_index'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(50.0))

df['prev_1c'] = df.groupby('馬名_clean')['first_corner_raw'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(8.0))
df['last_corner'] = df.groupby('馬名_clean')['last_corner_raw'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(8.0))
df['corner_diff'] = df.groupby('馬名_clean')['corner_diff_raw'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(0.0))

df['last_3f_avg_rank'] = df.groupby('馬名_clean')['last_3f'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(39.0))
df['avg_time_diff'] = df.groupby('馬名_clean')['time_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(1.5))
df['prev_date'] = df.groupby('馬名_clean')['date'].shift()
df['days_since_prev'] = (df['date'] - df['prev_date']).dt.days.fillna(14.0)
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()
df['prev_is_minami'] = df.groupby('馬名_clean')['is_minami_kanto'].shift().fillna(0).astype(int)

df['is_front_runner'] = (df['prev_1c'] <= 3.0).astype(int)
df['race_front_runners'] = df.groupby('race_id')['is_front_runner'].transform('sum')

# 🌟 新規追加2：展開の食い合いペナルティフラグ
# 自分が逃げ・先行で、なおかつレースに同型が3頭以上いるか[cite: 4]
df['high_pace_penalty'] = ((df['is_front_runner'] == 1) & (df['race_front_runners'] >= 3)).astype(int)

df['target_win'] = (df['target_rank_clean'] == 1.0).astype(int)
df['place_waku_combo'] = df['place_code'].astype(str) + "_" + df['waku_num'].astype(str)
df['trainer_clean'] = (df['調教師'] if '調教師' in df.columns else df['騎手']).astype(str)
df['jockey_trainer_combo'] = df['騎手'].astype(str) + "_" + df['trainer_clean']

def set_cumulative_win_rate(dataframe, group_col, out_col):
    runs = dataframe.groupby(group_col).cumcount()
    wins = dataframe.groupby(group_col)['target_win'].transform(lambda x: x.shift().cumsum().fillna(0))
    dataframe[out_col] = np.where(runs > 0, wins / runs, 0.05)

set_cumulative_win_rate(df, 'place_waku_combo', 'waku_win_rate')
set_cumulative_win_rate(df, 'trainer_clean', 'trainer_win_rate')
set_cumulative_win_rate(df, 'jockey_trainer_combo', 'combo_win_rate')
set_cumulative_win_rate(df, '騎手', 'jockey_win_rate')

# 🌟 ノイズ（平均着順系）を全削除し、新規データを追加
features = [
    'horse_prize_avg', 'race_prize_relative', 'race_prize_rank',
    'is_minami_kanto', 'prev_is_minami',
    # 削除: 'recent_avg_rank_3', 'recent_avg_rank_5', 'same_dist_avg_rank', 'same_place_avg_rank', 'bad_baba_avg_rank'
    'days_since_prev', 'is_large_weight_change',
    'prev_1c', 'last_corner', 'corner_diff', 'last_3f_avg_rank', 'avg_time_diff', 'is_bad_baba',
    'horse_career_runs', 'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    '斤量', 'body_weight', 'kinryo_weight_ratio', 'distance_num',
    'race_front_runners', 'waku_win_rate',
    'prev_time_index_avg', 'prev_start_index_avg', 'prev_last3f_index_avg', 'dist_change_num',
    'prev_class_weighted_score', 
    'prev_stall_rate', 'high_pace_penalty' # 🌟 追加した展開・なだれ込み評価
]

X = df[features].fillna(0.0).astype(float)
y_relevance = df['relevance']

groups = df.groupby('race_id', sort=False).size().values

print(f"✨ 全 {len(df)} 件 / {len(groups)} レースのグループ構造で 3連系特化モデル を学習します...")

def objective(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 15, 40), # 葉の数も少し絞って過学習防止
        'random_state': 42
    }
    train_size = int(len(X) * 0.8)
    group_cumsum = np.cumsum(groups)
    split_idx = np.searchsorted(group_cumsum, train_size)
    
    if split_idx == 0 or split_idx == len(groups):
        return 0.0
    
    train_groups = groups[:split_idx]
    valid_groups = groups[split_idx:]
    actual_train_size = int(np.sum(train_groups))
    
    X_train, y_train = X.iloc[:actual_train_size], y_relevance.iloc[:actual_train_size]
    X_valid, y_valid = X.iloc[actual_train_size:], y_relevance.iloc[actual_train_size:]

    model = lgb.LGBMRanker(**params)
    try:
        model.fit(
            X_train, y_train, group=train_groups,
            eval_set=[(X_valid, y_valid)], eval_group=[valid_groups],
            eval_at=[3], callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        return model.best_score_['valid_0']['ndcg@3']
    except Exception as e:
        return 0.0

print("\n--- 🔍 Optuna チューニング実行中 (LightGBM)... ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

best_params = study.best_params
best_params['objective'] = 'lambdarank'
best_params['metric'] = 'ndcg'
best_params['random_state'] = 42

print(f"✨ 最適パラメータ発見: {best_params}")

print("\n--- 【1/3】 LightGBM Ranker 本番学習中... ---")
ranker_lgb = lgb.LGBMRanker(**best_params)
ranker_lgb.fit(X, y_relevance, group=groups)

print("\n--- 【2/3】 XGBoost Ranker 学習中... ---")
ranker_xgb = xgb.XGBRanker(
    objective='rank:ndcg',
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=5,
    random_state=42
)
ranker_xgb.fit(X, y_relevance, group=groups)

print("\n--- 【3/3】 CatBoost Ranker 学習中... ---")
ranker_cat = cb.CatBoostRanker(
    loss_function='YetiRank',
    iterations=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    depth=5,
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

print(f"\n✨ 反映完了: 3連系特化Rankingモデル（{MODEL_FILE}）の出力が完了しました！")