import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import re

print("🌸 勝ち子ちゃん LGBMRankerモデル (フルスペック統合版)")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み、前処理を実行中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

# 1. 着順パース & ランキング評価（1着:3, 2着:2, 3着:1, 他:0）
def parse_rank(x):
    if pd.isna(x): return 99.0
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return 99.0

df['target_rank'] = df['着順'].apply(parse_rank) if '着順' in df.columns else df.get('target_rank').apply(parse_rank)
df = df[df['target_rank'] < 90.0].copy() # 正常着順のみ抽出
df['relevance'] = np.where(df['target_rank'] == 1.0, 3, np.where(df['target_rank'] == 2.0, 2, np.where(df['target_rank'] == 3.0, 1, 0)))

# 2. 基本列の数値化
df['first_corner'] = pd.to_numeric(df.get('first_corner'), errors='coerce').fillna(8.0)
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(0)
df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce').fillna(1400)
df['馬名_clean'] = df['馬名'].astype(str).apply(lambda x: re.sub(r'[\s\u3000]+', '', str(x)))

df['date'] = pd.to_datetime(df.get('date', pd.Series(['2020-01-01']*len(df))), errors='coerce')
df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

# 3. 性齢 & 馬体重パース（元のロジックを完全継承）
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

# 4. 過去平均成績 & 指数系（カンニング防止のshift）
df['recent_avg_rank'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(6.0)
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(8.0)

# 同距離適性 & カスタム指数
df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(df['recent_avg_rank'])
df['custom_time_index'] = 75.0 - (df['recent_avg_rank'].clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['prev_1c'].clip(upper=10.0)) * 6.5

# 単騎逃げ判定
df['is_nige_candidate'] = (df['prev_1c'] <= 2.5).astype(int)
nige_counts = df.groupby('race_id')['is_nige_candidate'].transform('sum')
df['single_escape_flag'] = np.where((df['is_nige_candidate'] == 1) & (nige_counts == 1), 1, 0)

# 騎手勝率
jockey_stats = (df[df['target_rank'] == 1.0].groupby('騎手')['target_rank'].count() / df.groupby('騎手')['target_rank'].count()).to_dict()
df['jockey_win_rate'] = df['騎手'].map(jockey_stats).fillna(0.05)

# 💡 全15項目のフルスペック特徴量リスト
features = [
    'recent_avg_rank', 'same_dist_avg_rank', 'prev_1c', 
    'custom_time_index', 'custom_start_index', 'single_escape_flag',
    '斤量', 'sex_code', 'age', 'body_weight', 'body_weight_diff',
    'kinryo_weight_ratio', 'jockey_win_rate', 'distance_num', '馬番_num'
]

# LGBMRankerグループ構築
df = df.sort_values('race_id').reset_index(drop=True)
groups = df.groupby('race_id').size().values

X = df[features].fillna(0)
y = df['relevance']

print(f"🧠 学習実行中 (LGBMRanker / 対象: {len(groups):,} レース / 特徴量: {len(features)}項目)...")

ranker = lgb.LGBMRanker(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)
ranker.fit(X, y, group=groups)

joblib.dump({
    'ranker_model': ranker,
    'model': ranker, # 互換用
    'features': features
}, MODEL_FILE)

print("✨ フルスペック×LGBMRankerモデル（`keiba_ai_model_nar.pkl`）の作成が完了しました！")