import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import re

print("🌸 勝ち子ちゃん 全データ100%活用・純粋能力特化モデル 学習スクリプト")

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

# 着順が存在する全レコードを1件も捨てずに使用
df = df[df['target_rank'].notna() & (df['target_rank'] < 90.0)].copy()

# 🎯 目的変数：3着以内（複勝圏）および 1着
df['target_place'] = (df['target_rank'] <= 3.0).astype(int)
df['target_win'] = (df['target_rank'] == 1.0).astype(int)

# 数値のパース処理
df['first_corner'] = pd.to_numeric(df.get('first_corner'), errors='coerce')
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(0)
df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce').fillna(1400)
df['馬名_clean'] = df['馬名'].astype(str).apply(lambda x: re.sub(r'[\s\u3000]+', '', str(x)))

df['date'] = pd.to_datetime(df.get('date', pd.Series(['2020-01-01']*len(df))), errors='coerce')
df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

# 性齢 & 馬体重
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
else:
    df['sex_code'], df['age'] = 0, 4.0

if '馬体重' in df.columns:
    parsed = df['馬体重'].apply(parse_weight_info)
    df['body_weight'], df['body_weight_diff'] = [x[0] for x in parsed], [x[1] for x in parsed]
else:
    df['body_weight'], df['body_weight_diff'] = 470.0, 0.0

df['kinryo_weight_ratio'] = df['斤量'] / df['body_weight'].clip(lower=350.0)

# 過去成績 & 指数系（オッズ・人気は完全排除）
df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['recent_avg_rank_5'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

# 💡 通算出走数のカウント（修正箇所）
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()

df['custom_time_index'] = 75.0 - (df['recent_avg_rank_3'].fillna(6.0).clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['prev_1c'].fillna(8.0).clip(upper=10.0)) * 6.5

# 騎手勝率
jockey_stats = (df[df['target_rank'] == 1.0].groupby('騎手')['target_rank'].count() / df.groupby('騎手')['target_rank'].count()).to_dict()
df['jockey_win_rate'] = df['騎手'].map(jockey_stats).fillna(0.05)

# 💡 オッズ・人気を排除した16項目の純粋能力特徴量
features = [
    'recent_avg_rank_3', 'recent_avg_rank_5', 'same_dist_avg_rank', 'prev_1c', 
    'horse_career_runs', 'custom_time_index', 'custom_start_index', 'jockey_win_rate', 
    '斤量', 'sex_code', 'age', 'body_weight', 'body_weight_diff', 
    'kinryo_weight_ratio', 'distance_num', '馬番_num'
]

X = df[features]
y_place = df['target_place']
y_win = df['target_win']

print(f"🧠 学習実行中 (総データ数: {len(df):,}件 / 特徴量: {len(features)}項目)...")

model_place = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42)
model_place.fit(X, y_place)

model_win = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42)
model_win.fit(X, y_win)

joblib.dump({
    'model_place': model_place,
    'model_win': model_win,
    'features': features
}, MODEL_FILE)

print("✨ 純粋能力学習モデル（`keiba_ai_model_nar.pkl`）の作成が完了しました！")