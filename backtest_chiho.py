import pandas as pd
import numpy as np
import joblib
import os
import re

print("🌸 勝ち子ちゃん バックテストツール (全特徴量拡張・3連単＆3連複対応版)")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE) or not os.path.exists(MODEL_FILE):
    print("⚠️ データまたはモデルが見つかりません。")
    exit()

print("📊 データを読み込み、前処理を実行中...")
df = pd.read_csv(CSV_FILE, low_memory=False)
model_data = joblib.load(MODEL_FILE)

model_win = model_data.get('model_win')
model_place = model_data.get('model_place')
features = model_data['features']

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

df['target_rank'] = df['着順'].apply(parse_rank).fillna(6.0) if '着順' in df.columns else df.get('target_rank').apply(parse_rank).fillna(6.0)
df = df[df['target_rank'] < 90.0].copy()

# 1. 数値・文字データの安全なパース
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

# 2. 南関東コード & 馬場状態
MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df['place_code'] = df['race_id'].astype(str).str[4:6]
df['is_minami_kanto'] = df['place_code'].isin(MINAMI_KANTO_CODES).astype(int)

baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df['baba_code'] = df.get('馬場', pd.Series(['良']*len(df))).map(baba_map).fillna(1)
df['is_bad_baba'] = (df['baba_code'] >= 3).astype(int)

# 3. 性齢 & 馬体重
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

# 4. 全過去走・調教師・コンビ・馬主統計データ復元
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

trainer_col = '調教師' if '調教師' in df.columns else '騎手'
df['trainer_clean'] = df[trainer_col].astype(str)
df['owner_clean'] = df['馬主'].astype(str) if '馬主' in df.columns else ''
df['jockey_trainer_combo'] = df['騎手'].astype(str) + "_" + df['trainer_clean']

trainer_stats = (df[df['target_rank'] == 1.0].groupby('trainer_clean')['target_rank'].count() / df.groupby('trainer_clean')['target_rank'].count()).to_dict()
combo_stats = (df[df['target_rank'] == 1.0].groupby('jockey_trainer_combo')['target_rank'].count() / df.groupby('jockey_trainer_combo')['target_rank'].count()).to_dict()
jockey_stats = (df[df['target_rank'] == 1.0].groupby('騎手')['target_rank'].count() / df.groupby('騎手')['target_rank'].count()).to_dict()
owner_stats = (df[df['target_rank'] == 1.0].groupby('owner_clean')['target_rank'].count() / df.groupby('owner_clean')['target_rank'].count()).to_dict() if '馬主' in df.columns else {}

df['trainer_win_rate'] = df['trainer_clean'].map(trainer_stats).fillna(0.05)
df['combo_win_rate'] = df['jockey_trainer_combo'].map(combo_stats).fillna(0.05)
df['jockey_win_rate'] = df['騎手'].map(jockey_stats).fillna(0.05)
df['owner_win_rate'] = df['owner_clean'].map(owner_stats).fillna(0.05)

waku_place_stats = (df[df['target_rank'] == 1.0].groupby(['place_code', 'distance_num', 'waku_num'])['target_rank'].count() / df.groupby(['place_code', 'distance_num', 'waku_num'])['target_rank'].count()).to_dict()
df['waku_place_win_rate'] = df.set_index(['place_code', 'distance_num', 'waku_num']).index.map(waku_place_stats).fillna(0.10)

# 5. AIモデル推論
X = df[features].copy()
if 'place_code' in X.columns:
    X['place_code'] = X['place_code'].astype('category')

print("🧠 最新モデルでAI確率を計算中...")
df['prob_win'] = model_win.predict_proba(X)[:, 1]
df['prob_place'] = model_place.predict_proba(X)[:, 1]

print("🏇 全22,408レースのシミュレーションを実行中...")

total_races = 0

# 3連単
hit_3tan_7pt = 0   # 7点
hit_3tan_8pt = 0   # 8点
hit_3tan_10pt = 0  # 10点
hit_3tan_12pt = 0  # 12点

# 3連複
hit_3fuku_4box = 0   # 4頭BOX (4点)
hit_3fuku_1jiku4 = 0 # 1軸4頭流し (6点)

for race_id, group in df.groupby('race_id'):
    if len(group) < 8 or not (1.0 in group['target_rank'].values):
        continue

    total_races += 1

    sorted_group = group.sort_values(by=['prob_place', 'prob_win'], ascending=[False, False])
    ranks = sorted_group['target_rank'].values
    if len(ranks) < 6: continue
    
    r1, r2, r3, r4, r5, r6 = ranks[0], ranks[1], ranks[2], ranks[3], ranks[4], ranks[5]

    # --- 3連単 検証 ---
    if (1.0 in {r1, r2} and 2.0 in {r1, r2}) and (3.0 in {r3, r4, r5, r6}):
        hit_3tan_7pt += 1

    # 8点買い (1着:1位 / 2着:1~3位 / 3着:1~6位)
    if (1.0 in {r1, r2}) and (2.0 in {r1, r2, r3}) and (3.0 in {r2, r3, r4, r5}) and len({r1,r2,r3}) >= 2:
        hit_3tan_8pt += 1

    if (1.0 in {r1, r2}) and (2.0 in {r1, r2, r3}) and (3.0 in {r1, r2, r3, r4, r5}):
        hit_3tan_10pt += 1

    if (r1 == 1.0) and (2.0 in {r2, r3, r4}) and (3.0 in {r2, r3, r4, r5}):
        hit_3tan_12pt += 1

    # --- 3連複 検証 ---
    if {1.0, 2.0, 3.0}.issubset({r1, r2, r3, r4}):
        hit_3fuku_4box += 1

    if (r1 in {1.0, 2.0, 3.0}) and {1.0, 2.0, 3.0}.issubset({r1, r2, r3, r4, r5}):
        hit_3fuku_1jiku4 += 1

print("\n" + "="*60)
print("🎯 地方競馬 全過去データ(22,408レース) バックテスト結果")
print("="*60)
print(f"🔹 検証対象レース数 : {total_races:,} レース")
print("-" * 60)
print("【 3連単 券種 】")
print(f"🔥  7点買い (1,2 - 1,2 - 3,4,5,6)  : {hit_3tan_7pt / total_races * 100:.1f}%  ({hit_3tan_7pt:,}回的中)")
print(f"🔥  8点買い (1,2 - 1,2,3 - 2,3,4,5): {hit_3tan_8pt / total_races * 100:.1f}%  ({hit_3tan_8pt:,}回的中)")
print(f"🔥 10点買い (1,2 - 1,2,3 - 1~5)    : {hit_3tan_10pt / total_races * 100:.1f}%  ({hit_3tan_10pt:,}回的中)")
print(f"🔥 12点買い (1固定 → 2,3,4 → 2~5)  : {hit_3tan_12pt / total_races * 100:.1f}%  ({hit_3tan_12pt:,}回的中)")
print("-" * 60)
print("【 3連複 券種 】")
print(f"🌸  4頭BOX (上位4頭 / 4点)          : {hit_3fuku_4box / total_races * 100:.1f}%  ({hit_3fuku_4box:,}回的中)")
print(f"🌸  1軸4頭流し (1位軸 ＝ 2~5位 / 6点): {hit_3fuku_1jiku4 / total_races * 100:.1f}%  ({hit_3fuku_1jiku4:,}回的中)")
print("="*60)