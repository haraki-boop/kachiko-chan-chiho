
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import re
import optuna
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("🌸 勝ち子ちゃん 純粋確率モデル (Optuna最適化版) 学習スクリプト")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み、前処理を実行中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

df['target_rank'] = df['着順'].apply(parse_rank) if '着順' in df.columns else df.get('target_rank').apply(parse_rank)
df = df[df['target_rank'].notna() & (df['target_rank'] < 90.0)].copy()

# ターゲット設定
df['target_rentai'] = (df['target_rank'] <= 2.0).astype(int)
df['target_win'] = (df['target_rank'] == 1.0).astype(int)

# 前処理群
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

def parse_sex_age(val):
    if pd.isna(val): return 0, 4.0
    s = str(val).strip()
    sex_code = 1 if '牝' in s else (2 if 'セ' in s else 0)
    age_match = re.search(r'\d+', s)
    age = float(age_match.group(0)) if age_match else 4.0
    return sex_code, age

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0) if m else (470.0, 0.0)

if '性齢' in df.columns:
    parsed = df['性齢'].apply(parse_sex_age)
    df['sex_code'], df['age'] = [x[0] for x in parsed], [x[1] for x in parsed]
else: df['sex_code'], df['age'] = 0, 4.0

if '馬体重' in df.columns:
    parsed = df['馬体重'].apply(parse_weight_info)
    df['body_weight'], df['body_weight_diff'] = [x[0] for x in parsed], [x[1] for x in parsed]
else: df['body_weight'], df['body_weight_diff'] = 470.0, 0.0

df['kinryo_weight_ratio'] = df['斤量'] / df['body_weight'].clip(lower=350.0)
df['is_large_weight_change'] = (df['body_weight_diff'].abs() >= 10.0).astype(int)

baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df['baba_code'] = df.get('馬場', pd.Series(['良']*len(df))).map(baba_map).fillna(1)
df['is_bad_baba'] = (df['baba_code'] >= 3).astype(int)

# 過去走集計
df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['recent_avg_rank_5'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['last_3f_avg_rank'] = df.groupby('馬名_clean')['last_3f'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(39.0))
df['avg_time_diff'] = df.groupby('馬名_clean')['time_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(1.5))

df['bad_baba_avg_rank'] = df[df['is_bad_baba'] == 1].groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['bad_baba_avg_rank'] = df.groupby('馬名_clean')['bad_baba_avg_rank'].ffill().fillna(df['recent_avg_rank_3'])

df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['same_place_avg_rank'] = df.groupby(['馬名_clean', 'place_code'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['prev_date'] = df.groupby('馬名_clean')['date'].shift()
df['days_since_prev'] = (df['date'] - df['prev_date']).dt.days.fillna(30.0)
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()
df['prev_is_minami'] = df.groupby('馬名_clean')['is_minami_kanto'].shift().fillna(0).astype(int)

df['custom_time_index'] = 75.0 - (df['recent_avg_rank_3'].fillna(6.0).clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['prev_1c'].fillna(8.0).clip(upper=10.0)) * 6.5

# 未来データのリーク防止: 全体の平均ではなく、過去の累積平均を使用
df['trainer_clean'] = df.get('調教師', df.get('騎手')).astype(str)
df['jockey_clean'] = df.get('騎手', '').astype(str)
df['jockey_trainer_combo'] = df['jockey_clean'] + "_" + df['trainer_clean']

# リークしない勝率計算 (Expanding Mean: 過去走のみの累積平均)
def calc_expanding_win_rate(group_col):
    df_sorted = df.sort_values(['date', 'race_id'])
    return df_sorted.groupby(group_col)['target_win'].transform(lambda x: x.shift().expanding().mean()).fillna(0.05)

df['trainer_win_rate'] = calc_expanding_win_rate('trainer_clean')
df['jockey_win_rate'] = calc_expanding_win_rate('jockey_clean')
df['combo_win_rate'] = calc_expanding_win_rate('jockey_trainer_combo')

features = [
    'is_minami_kanto', 'prev_is_minami', 'recent_avg_rank_3', 'recent_avg_rank_5', 
    'same_dist_avg_rank', 'same_place_avg_rank', 'days_since_prev', 'is_large_weight_change',
    'prev_1c', 'last_corner', 'corner_diff', 'last_3f_avg_rank', 'avg_time_diff', 'bad_baba_avg_rank', 'is_bad_baba',
    'horse_career_runs', 'custom_time_index', 'custom_start_index', 
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    '斤量', 'body_weight', 'kinryo_weight_ratio', 'distance_num'
]

X = df[features].fillna(0.0)
y_rentai = df['target_rentai']
y_win = df['target_win']

# --------------------------------------------------------
# Optuna 自動ハイパーパラメータ最適化関数
# --------------------------------------------------------
def optimize_lgbm(X_data, y_data, target_name):
    print(f"🔍 Optunaで {target_name} モデルのハイパーパラメータを自動チューニング中...")
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 7, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_data):
            X_tr, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
            y_tr, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_val)[:, 1]
            
            # LogLoss計算
            eps = 1e-15
            preds = np.clip(preds, eps, 1 - eps)
            loss = -np.mean(y_val * np.log(preds) + (1 - y_val) * np.log(1 - preds))
            scores.append(loss)
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print(f"✨ {target_name} の最適化完了 Best Score: {study.best_value:.4f}")
    return study.best_params

# 連対モデルの最適化＆学習
best_params_rentai = optimize_lgbm(X, y_rentai, "連対(2着以内)")
model_place = lgb.LGBMClassifier(**best_params_rentai, random_state=42)
model_place.fit(X, y_rentai)

# 勝利モデルの最適化＆学習
best_params_win = optimize_lgbm(X, y_win, "勝利(1着)")
model_win = lgb.LGBMClassifier(**best_params_win, random_state=42)
model_win.fit(X, y_win)

joblib.dump({
    'model_place': model_place,
    'model_win': model_win,
    'features': features
}, MODEL_FILE)

print("✨ 自動チューニング済みモデル（`keiba_ai_model_nar.pkl`）の保存が完了しました！")
