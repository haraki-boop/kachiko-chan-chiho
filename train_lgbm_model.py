import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import re

print("🌸 勝ち子ちゃん 全29特徴量・完全能力モデル 学習スクリプト")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み、拡張前処理を実行中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

df['target_rank'] = df['着順'].apply(parse_rank) if '着順' in df.columns else df.get('target_rank').apply(parse_rank)
df = df[df['target_rank'].notna() & (df['target_rank'] < 90.0)].copy()

df['target_place'] = (df['target_rank'] <= 3.0).astype(int)
df['target_win'] = (df['target_rank'] == 1.0).astype(int)

# 基本変換
df['first_corner'] = pd.to_numeric(df.get('first_corner'), errors='coerce')
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
else: df['sex_code'], df['age'] = 0, 4.0

if '馬体重' in df.columns:
    parsed = df['馬体重'].apply(parse_weight_info)
    df['body_weight'], df['body_weight_diff'] = [x[0] for x in parsed], [x[1] for x in parsed]
else: df['body_weight'], df['body_weight_diff'] = 470.0, 0.0

df['kinryo_weight_ratio'] = df['斤量'] / df['body_weight'].clip(lower=350.0)
df['is_large_weight_change'] = (df['body_weight_diff'].abs() >= 10.0).astype(int)

# 馬場状態パース (1:良, 2:稍重, 3:重, 4:不良)
baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df['baba_code'] = df.get('馬場', pd.Series(['良']*len(df))).map(baba_map).fillna(1)
df['is_bad_baba'] = (df['baba_code'] >= 3).astype(int)

# 過去走・ローテーション・走りの質データ
df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['recent_avg_rank_5'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

# 🆕 新規項目：上がり3F順位 & 勝ち馬からの平均タイム差
df['last_3f_avg_rank'] = df.groupby('馬名_clean')['last_3f'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(39.0))
df['avg_time_diff'] = df.groupby('馬名_clean')['time_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(1.5))

# 🆕 新規項目：道悪（重・不良）限定の平均着順
df['bad_baba_avg_rank'] = df[df['is_bad_baba'] == 1].groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['bad_baba_avg_rank'] = df.groupby('馬名_clean')['bad_baba_avg_rank'].ffill().fillna(df['recent_avg_rank_3'])

# 同コース・間隔
df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['same_place_avg_rank'] = df.groupby(['馬名_clean', 'place_code'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['prev_date'] = df.groupby('馬名_clean')['date'].shift()
df['days_since_prev'] = (df['date'] - df['prev_date']).dt.days.fillna(30.0)
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()
df['prev_is_minami'] = df.groupby('馬名_clean')['is_minami_kanto'].shift().fillna(0).astype(int)

# 指数
df['custom_time_index'] = 75.0 - (df['recent_avg_rank_3'].fillna(6.0).clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['prev_1c'].fillna(8.0).clip(upper=10.0)) * 6.5

# 🆕 新規項目：調教師勝率・騎手×調教師コンビ勝率
trainer_col = '調教師' if '調教師' in df.columns else '騎手'
df['trainer_clean'] = df[trainer_col].astype(str)
df['jockey_trainer_combo'] = df['騎手'].astype(str) + "_" + df['trainer_clean']

trainer_stats = (df[df['target_rank'] == 1.0].groupby('trainer_clean')['target_rank'].count() / df.groupby('trainer_clean')['target_rank'].count()).to_dict()
combo_stats = (df[df['target_rank'] == 1.0].groupby('jockey_trainer_combo')['target_rank'].count() / df.groupby('jockey_trainer_combo')['target_rank'].count()).to_dict()
jockey_stats = (df[df['target_rank'] == 1.0].groupby('騎手')['target_rank'].count() / df.groupby('騎手')['target_rank'].count()).to_dict()

df['trainer_win_rate'] = df['trainer_clean'].map(trainer_stats).fillna(0.05)
df['combo_win_rate'] = df['jockey_trainer_combo'].map(combo_stats).fillna(0.05)
df['jockey_win_rate'] = df['騎手'].map(jockey_stats).fillna(0.05)

# 🆕 新規項目：コース×距離×枠順の構造的勝率
waku_place_stats = (df[df['target_rank'] == 1.0].groupby(['place_code', 'distance_num', 'waku_num'])['target_rank'].count() / df.groupby(['place_code', 'distance_num', 'waku_num'])['target_rank'].count()).to_dict()
df['waku_place_win_rate'] = df.set_index(['place_code', 'distance_num', 'waku_num']).index.map(waku_place_stats).fillna(0.10)

# 💡 全29の特徴量
features = [
    'is_minami_kanto', 'prev_is_minami', 'recent_avg_rank_3', 'recent_avg_rank_5', 
    'same_dist_avg_rank', 'same_place_avg_rank', 'days_since_prev', 'is_large_weight_change',
    'prev_1c', 'last_3f_avg_rank', 'avg_time_diff', 'bad_baba_avg_rank', 'is_bad_baba',
    'horse_career_runs', 'custom_time_index', 'custom_start_index', 
    'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate', 'waku_place_win_rate',
    '斤量', 'sex_code', 'age', 'body_weight', 'body_weight_diff', 
    'kinryo_weight_ratio', 'distance_num', '馬番_num', 'place_code'
]

X = df[features].copy()
X['place_code'] = X['place_code'].astype('category')

y_place = df['target_place']
y_win = df['target_win']

print(f"🧠 AIモデル学習中 (総データ数: {len(df):,}件 / 評価特徴量: {len(features)}項目)...")

model_place = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42)
model_place.fit(X, y_place)

model_win = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, random_state=42)
model_win.fit(X, y_win)

joblib.dump({
    'model_place': model_place,
    'model_win': model_win,
    'features': features
}, MODEL_FILE)

print("✨ 全29特徴量モデル（`keiba_ai_model_nar.pkl`）の学習が完了しました！")