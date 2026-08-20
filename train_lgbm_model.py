import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import os
import re

print("🤖 勝ち子ちゃん AI機械学習スクリプト (Step 1: 安全＆新項目対応版)")
CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ エラー: {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み中...")
# 混合型エラー対策
df = pd.read_csv(CSV_FILE, low_memory=False)

# 1. 基礎項目の安全な数値化
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(0)
df['first_corner'] = pd.to_numeric(df.get('first_corner'), errors='coerce').fillna(8.0)

# 着順の安全変換
rank_series = pd.to_numeric(df.get('target_rank', df.get('着順')), errors='coerce')
df['target_rank'] = rank_series.fillna(6.0)

# 2. 拡張項目（性齢・馬体重など）が存在すれば安全に前処理
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
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return 470.0, 0.0

if '性齢' in df.columns:
    sex_age_parsed = df['性齢'].apply(parse_sex_age)
    df['sex_code'] = [x[0] for x in sex_age_parsed]
    df['age'] = [x[1] for x in sex_age_parsed]
else:
    df['sex_code'], df['age'] = 0, 4.0

if '馬体重' in df.columns:
    weight_parsed = df['馬体重'].apply(parse_weight_info)
    df['body_weight'] = [x[0] for x in weight_parsed]
    df['body_weight_diff'] = [x[1] for x in weight_parsed]
else:
    df['body_weight'], df['body_weight_diff'] = 470.0, 0.0

# 3. 独自指数の算出
df['custom_time_index'] = 75.0 - (df['target_rank'].clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['first_corner'].clip(upper=10.0)) * 6.5
df['kinryo_weight_ratio'] = df['斤量'] / df['body_weight'].clip(lower=350.0)

# 4. 特徴量リスト（オッズ・人気は除外）
features = [
    '斤量', '馬番_num', 'first_corner', 'custom_time_index', 'custom_start_index',
    'sex_code', 'age', 'body_weight', 'body_weight_diff', 'kinryo_weight_ratio'
]

X = df[features].fillna(0)
y = (df['target_rank'] == 1).astype(int)

print(f"🧬 使用する比較項目数: {len(features)} 項目")
print("🧠 AIモデルの学習を開始...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=6, random_state=42)
model.fit(X_train, y_train)

joblib.dump({'model': model, 'features': features}, MODEL_FILE)
print(f"✨ エラーなしで学習成功！モデルを保存しました: {MODEL_FILE}")