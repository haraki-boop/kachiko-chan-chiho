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

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("🚀 勝ち子ちゃん 第3形態【3連系特化・アンサンブル学習版】")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み、前処理を実行中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

# 🔥 バグ修正: 文字（降着・取消など）が混じった着順データを確実に数値化
def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

# 正解ラベル（今回のレースの着順）
target_col = 'target_rank' if 'target_rank' in df.columns else '着順'
df['target_rank_clean'] = df[target_col].apply(parse_rank)
df = df[df['target_rank_clean'].notna() & (df['target_rank_clean'] < 90.0)].copy()

df['target_rentai'] = (df['target_rank_clean'] <= 2.0).astype(int)
df['target_win'] = (df['target_rank_clean'] == 1.0).astype(int)

df['first_corner'] = pd.to_numeric(df.get('first_corner', df.get('1角')), errors='coerce').fillna(8.0)
df['last_corner'] = pd.to_numeric(df.get('last_corner', df.get('4角')), errors='coerce').fillna(df['first_corner'])
df['corner_diff'] = df['first_corner'] - df['last_corner']
df['last_3f'] = pd.to_numeric(df.get('last_3f', df.get('上り')), errors='coerce')
df['time_diff'] = pd.to_numeric(df.get('time_diff', df.get('着差')), errors='coerce').fillna(1.5)
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(0)
df['waku_num'] = pd.to_numeric(df.get('枠番'), errors='coerce').fillna(0)
df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce').fillna(1400)
df['馬名_clean'] = df['馬名'].astype(str).apply(lambda x: re.sub(r'[\s\u3000]+', '', str(x)))

df['date'] = pd.to_datetime(df.get('date', pd.Series(['2020-01-01']*len(df))), errors='coerce')
df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

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

# 🔥 バグ修正: target_rank_clean（数値化済み）を使って過去成績を計算する
df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(5.0))
df['recent_avg_rank_5'] = df.groupby('馬名_clean')['target_rank_clean'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean().fillna(5.0))
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(8.0))
df['last_3f_avg_rank'] = df.groupby('馬名_clean')['last_3f'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(39.0))
df['avg_time_diff'] = df.groupby('馬名_clean')['time_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(1.5))
df['bad_baba_avg_rank'] = df[df['is_bad_baba'] == 1].groupby('馬名_clean')['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['bad_baba_avg_rank'] = df.groupby('馬名_clean')['bad_baba_avg_rank'].ffill().fillna(df['recent_avg_rank_3'])
df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(5.0))
df['same_place_avg_rank'] = df.groupby(['馬名_clean', 'place_code'])['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(5.0))
df['prev_date'] = df.groupby('馬名_clean')['date'].shift()
df['days_since_prev'] = (df['date'] - df['prev_date']).dt.days.fillna(14.0)
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()
df['prev_is_minami'] = df.groupby('馬名_clean')['is_minami_kanto'].shift().fillna(0).astype(int)

df['custom_time_index'] = 75.0 - (df['recent_avg_rank_3'].clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['prev_1c'].clip(upper=10.0)) * 6.5

df['is_front_runner'] = (df['prev_1c'] <= 3.0).astype(int)
df['race_front_runners'] = df.groupby('race_id')['is_front_runner'].transform('sum')

df['place_waku_combo'] = df['place_code'].astype(str) + "_" + df['waku_num'].astype(str)
waku_stats = (df[df['target_rank_clean'] == 1.0].groupby('place_waku_combo')['target_rank_clean'].count() / df.groupby('place_waku_combo')['target_rank_clean'].count()).to_dict()
df['waku_win_rate'] = df['place_waku_combo'].map(waku_stats).fillna(0.05)

trainer_col = '調教師' if '調教師' in df.columns else '騎手'
df['trainer_clean'] = df[trainer_col].astype(str)
df['jockey_trainer_combo'] = df['騎手'].astype(str) + "_" + df['trainer_clean']
trainer_stats = (df[df['target_rank_clean'] == 1.0].groupby('trainer_clean')['target_rank_clean'].count() / df.groupby('trainer_clean')['target_rank_clean'].count()).to_dict()
combo_stats = (df[df['target_rank_clean'] == 1.0].groupby('jockey_trainer_combo')['target_rank_clean'].count() / df.groupby('jockey_trainer_combo')['target_rank_clean'].count()).to_dict()
jockey_stats = (df[df['target_rank_clean'] == 1.0].groupby('騎手')['target_rank_clean'].count() / df.groupby('騎手')['target_rank_clean'].count()).to_dict()
df['trainer_win_rate'] = df['trainer_clean'].map(trainer_stats).fillna(0.05)
df['combo_win_rate'] = df['jockey_trainer_combo'].map(combo_stats).fillna(0.05)
df['jockey_win_rate'] = df['騎手'].map(jockey_stats).fillna(0.05)

# 🌟 アプリ側（predict_with_gemini_chiho.py）と完全に一致させた27個の特徴量リスト
features = [
    'is_minami_kanto', 'prev_is_minami', 'recent_avg_rank_3', 'recent_avg_rank_5', 
    'same_dist_avg_rank', 'same_place_avg_rank', 'days_since_prev', 'is_large_weight_change',
    'prev_1c', 'last_corner', 'corner_diff', 'last_3f_avg_rank', 'avg_time_diff', 'bad_baba_avg_rank', 'is_bad_baba',
    'horse_career_runs', 'custom_time_index', 'custom_start_index', 
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    '斤量', 'body_weight', 'kinryo_weight_ratio', 'distance_num',
    'race_front_runners', 'waku_win_rate'
]

X = df[features].fillna(0.0).astype(float)
y_rentai = df['target_rentai']
y_win = df['target_win']

# ========================================================
# 🤖 アンサンブル用：各AIのOptuna最適化ロジック
# ========================================================
n_trials_opt = 20

def optimize_lgb(X_data, y_data, target_name):
    print(f"🔍 [1/3] LightGBM: {target_name} モデルをチューニング中...")
    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 7)
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_data):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_data.iloc[train_idx], y_data.iloc[train_idx])
            preds = np.clip(model.predict_proba(X_data.iloc[val_idx])[:, 1], 1e-15, 1 - 1e-15)
            scores.append(-np.mean(y_data.iloc[val_idx] * np.log(preds) + (1 - y_data.iloc[val_idx]) * np.log(1 - preds)))
        return np.mean(scores)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials_opt)
    return lgb.LGBMClassifier(**study.best_params, random_state=42).fit(X_data, y_data)

def optimize_xgb(X_data, y_data, target_name):
    print(f"🔍 [2/3] XGBoost: {target_name} モデルをチューニング中...")
    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss', 'verbosity': 0, 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7)
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_data):
            model = xgb.XGBClassifier(**params)
            model.fit(X_data.iloc[train_idx], y_data.iloc[train_idx])
            preds = np.clip(model.predict_proba(X_data.iloc[val_idx])[:, 1], 1e-15, 1 - 1e-15)
            scores.append(-np.mean(y_data.iloc[val_idx] * np.log(preds) + (1 - y_data.iloc[val_idx]) * np.log(1 - preds)))
        return np.mean(scores)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials_opt)
    return xgb.XGBClassifier(**study.best_params, random_state=42).fit(X_data, y_data)

def optimize_cat(X_data, y_data, target_name):
    print(f"🔍 [3/3] CatBoost: {target_name} モデルをチューニング中...")
    def objective(trial):
        params = {
            'loss_function': 'Logloss', 'verbose': False, 'random_state': 42,
            'iterations': trial.suggest_int('iterations', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'depth': trial.suggest_int('depth', 3, 7)
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_data):
            model = cb.CatBoostClassifier(**params)
            model.fit(X_data.iloc[train_idx], y_data.iloc[train_idx])
            preds = np.clip(model.predict_proba(X_data.iloc[val_idx])[:, 1], 1e-15, 1 - 1e-15)
            scores.append(-np.mean(y_data.iloc[val_idx] * np.log(preds) + (1 - y_data.iloc[val_idx]) * np.log(1 - preds)))
        return np.mean(scores)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials_opt)
    return cb.CatBoostClassifier(**study.best_params, random_state=42, verbose=False).fit(X_data, y_data)

# ========================================================
# ⚔️ 3つのAIを同時に学習して合体
# ========================================================
print("\n--- 【連対率(2着以内) モデルの学習】 ---")
model_place_lgb = optimize_lgb(X, y_rentai, "連対(LGBM)")
model_place_xgb = optimize_xgb(X, y_rentai, "連対(XGBoost)")
model_place_cat = optimize_cat(X, y_rentai, "連対(CatBoost)")

print("\n--- 【勝率(1着) モデルの学習】 ---")
model_win_lgb = optimize_lgb(X, y_win, "勝利(LGBM)")
model_win_xgb = optimize_xgb(X, y_win, "勝利(XGBoost)")
model_win_cat = optimize_cat(X, y_win, "勝利(CatBoost)")

joblib.dump({
    'model_place_lgb': model_place_lgb,
    'model_win_lgb': model_win_lgb,
    'model_place_xgb': model_place_xgb,
    'model_win_xgb': model_win_xgb,
    'model_place_cat': model_place_cat,
    'model_win_cat': model_win_cat,
    'features': features
}, MODEL_FILE)

print(f"\n✨ 限界突破！3つのAIを統合したファイル（{MODEL_FILE}）の保存が完了しました！")