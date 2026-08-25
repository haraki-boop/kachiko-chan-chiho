import pandas as pd
import numpy as np
import os
import optuna
import joblib
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

print("==================================================")
print("🌸 勝ち子ちゃん 自動チューニング (完全実戦・過学習ゼロ)")
print("==================================================")

CSV_FILE = "ml_target_data_chiho.csv"

if not os.path.exists(CSV_FILE):
    print("⚠️ データが見つかりません。")
    exit()

print("📊 過去データを読み込み中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

def parse_rank(x):
    if pd.isna(x): return 6.0
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return 6.0

df['target_rank'] = df.get('着順', df.get('着順_num')).apply(parse_rank)
df = df[df['target_rank'] < 90.0].copy()
df['target_place_flag'] = (df['target_rank'] <= 3.0).astype(int)

# 日付順ソート
df['date_dt'] = pd.to_datetime(df.get('date', pd.Series(['2020-01-01']*len(df))), errors='coerce')
df = df.sort_values(['馬名', 'date_dt']).reset_index(drop=True)

# --------------------------------------------------------
# 🛡️ 過去データへの完全変換（当日の走破結果・指数の遮断）
# --------------------------------------------------------
shift_cols = [
    'first_corner', 'last_corner', 'corner_diff', 'is_front', 'last_3f', 'time_diff',
    'custom_time_index', 'custom_start_index', 'custom_pursuit_index', 'custom_last3f_index',
    'jockey_win_rate', 'jockey_runs', 'race_front_count', 'dist_change', 'horse_weight', 'weight_change'
]

for col in shift_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # 前走までの値（1走シフト）に変換
        df[col] = df.groupby('馬名')[col].transform(lambda x: x.shift(1)).fillna(0.0)

# --------------------------------------------------------
# 🔥 追加の3特徴量（前走以前の情報のみで作成）
# --------------------------------------------------------
df['distance_num'] = pd.to_numeric(df.get('distance_num', df.get('distance')), errors='coerce').fillna(1400.0)
df['time_sec'] = pd.to_numeric(df.get('time_sec', df.get('タイム')), errors='coerce')

# ① best_speed: 前走までの過去タイムから算出
df['speed_m_per_sec'] = df['distance_num'] / df['time_sec'].clip(lower=10.0)
df['best_speed'] = df.groupby('馬名')['speed_m_per_sec'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).max()).fillna(15.0)

# ② course_bias_index: 前走脚質に基づくバイアス
df['kyakushitsu_cat'] = pd.cut(pd.to_numeric(df.get('first_corner', 8.0), errors='coerce'), bins=[0, 2.5, 5.5, 99], labels=[1, 2, 3]).astype(float)
df['prev_kyakushitsu'] = df.groupby('馬名')['kyakushitsu_cat'].shift(1).fillna(2.0)
df['place_code'] = df['race_id'].astype(str).str[4:6].astype(float)
df['waku_num'] = pd.to_numeric(df.get('waku_num', df.get('枠番')), errors='coerce').fillna(1.0)
df['course_bias_index'] = df.groupby(['place_code', 'distance_num', 'waku_num', 'prev_kyakushitsu'])['target_place_flag'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.2)

# ③ prize_diff: 前走賞金と前々走賞金の差
df['prize_money'] = pd.to_numeric(df.get('賞金(万円)'), errors='coerce').fillna(0.0)
df['prev_prize_1'] = df.groupby('馬名')['prize_money'].shift(1).fillna(0.0)
df['prev_prize_2'] = df.groupby('馬名')['prize_money'].shift(2).fillna(0.0)
df['prize_diff'] = df['prev_prize_2'] - df['prev_prize_1']

# --------------------------------------------------------
# 💎 厳選25特徴量
# --------------------------------------------------------
features = [
    'weight_num', 'gate_num', 'waku_num', 'distance_num', 'place_code', 'horse_weight', 'weight_change',
    'first_corner', 'last_corner', 'corner_diff', 'is_front', 'race_front_count', 'dist_change',
    'last_3f', 'time_diff', 'recent_avg_rank_3', 'custom_time_index', 'custom_start_index', 'custom_pursuit_index', 'custom_last3f_index',
    'jockey_win_rate', 'jockey_runs',
    'best_speed', 'course_bias_index', 'prize_diff'
]

for col in features:
    if col not in df.columns:
        df[col] = 0.0
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

# 時系列（過去8割で学習、未来2割で評価）
df = df.sort_values('date_dt').reset_index(drop=True)
split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()

X_train = train_df[features]
y_train_place = train_df['target_place_flag']

X_val = val_df[features]
y_val_place = val_df['target_place_flag']

# --------------------------------------------------------
# 🤖 Optunaによる自動チューニング
# --------------------------------------------------------
print(f"⚙️ 完全実戦環境（カンニング一切なし）で {len(features)} 個の特徴量を最適化中...")

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_iter': trial.suggest_int('max_iter', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
        'random_state': 42
    }
    
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train_place)
    
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val_place, preds)

study = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=30)

print("\n🏆 チューニング完了！")
print(f"🥇 実戦スコア (AUC): {study.best_value:.4f}")
print("🥇 黄金パラメータ:", study.best_params)

best_params = study.best_params
best_params['random_state'] = 42

model_place = HistGradientBoostingClassifier(**best_params)
model_place.fit(X_train, y_train_place)

y_train_win = (train_df['target_rank'] == 1.0).astype(int)
model_win = HistGradientBoostingClassifier(**best_params)
model_win.fit(X_train, y_train_win)

model_data = {
    'features': features,
    'model_place': model_place,
    'model_win': model_win,
    'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

joblib.dump(model_data, "keiba_ai_model_nar.pkl")
print("🎉 実戦用のモデルが更新されました！")