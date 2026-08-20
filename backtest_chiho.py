import pandas as pd
import numpy as np
import joblib
import os
import re

print("🌸 勝ち子ちゃん バックテストツール (3連単 7点〜12点検証版)")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE) or not os.path.exists(MODEL_FILE):
    print("⚠️ データまたはモデルが見つかりません。")
    exit()

print("📊 データを読み込み、新AIモデルをセットアップ中...")
df = pd.read_csv(CSV_FILE, low_memory=False)
model_data = joblib.load(MODEL_FILE)

model_win = model_data.get('model_win', model_data.get('model'))
model_place = model_data.get('model_place', model_data.get('model'))
features = model_data['features']

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

df['target_rank'] = df['着順'].apply(parse_rank).fillna(6.0) if '着順' in df.columns else df.get('target_rank').apply(parse_rank).fillna(6.0)
df['first_corner'] = pd.to_numeric(df.get('first_corner'), errors='coerce').fillna(8.0)
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(0)
df['馬名_clean'] = df['馬名'].astype(str).apply(lambda x: re.sub(r'[\s\u3000]+', '', x))

df['date'] = pd.to_datetime(df.get('date', pd.Series(['2020-01-01']*len(df))), errors='coerce')
df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

df['recent_avg_rank'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(6.0)
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(8.0)

df_sim = df.copy()
df_sim['custom_time_index'] = 75.0 - (df_sim['recent_avg_rank'].clip(1, 14) - 3.0) * 3.5 + (df_sim['斤量'] - 54.0) * 1.5
df_sim['custom_start_index'] = (12.0 - df_sim['prev_1c'].clip(upper=10.0)) * 6.5

for col in features:
    if col not in df_sim.columns: df_sim[col] = 0.0

X = df_sim[features].fillna(0)

# 1着確率 & 3着内確率
df_sim['prob_win'] = model_win.predict_proba(X)[:, 1] if hasattr(model_win, "predict_proba") else model_win.predict(X)
df_sim['prob_place'] = model_place.predict_proba(X)[:, 1] if hasattr(model_place, "predict_proba") else model_place.predict(X)

print("🏇 全レースシミュレーションを実行中...")

total_races = 0
hit_7pt = 0   # 7点買い
hit_8pt = 0   # 8点買い
hit_10pt = 0  # 10点買い
hit_12pt = 0  # 12点買い

for race_id, group in df_sim.groupby('race_id'):
    if len(group) < 8 or not (1.0 in group['target_rank'].values):
        continue

    group = group.copy()
    
    t_min, t_max = group['custom_time_index'].min(), group['custom_time_index'].max()
    s_min, s_max = group['custom_start_index'].min(), group['custom_start_index'].max()
    time_norm = (group['custom_time_index'] - t_min) / (t_max - t_min + 1e-5) if t_max > t_min else 0.0
    start_norm = (group['custom_start_index'] - s_min) / (s_max - s_min + 1e-5) if s_max > s_min else 0.0
    
    group['first_corner_rank'] = group['prev_1c'].rank(ascending=True, method='min')
    priority = np.where(group['first_corner_rank'] <= 2, 0.25, 0.0)
    
    group['axis_score'] = (group['prob_win'] * 0.4) + (time_norm * 0.3) + (start_norm * 0.1) + priority
    group['himo_score'] = (group['prob_place'] * 0.4) + (time_norm * 0.3) + (start_norm * 0.2)

    total_races += 1

    # 上位6頭の着順を取得 (1位〜6位)
    sorted_group = group.sort_values(by=['axis_score', 'recent_avg_rank'], ascending=[False, True])
    ranks = sorted_group['target_rank'].values
    if len(ranks) < 6: continue
    
    r1, r2, r3, r4, r5, r6 = ranks[0], ranks[1], ranks[2], ranks[3], ranks[4], ranks[5]

    # --- ① 7点買い: 1,2 - 1,2 - 3,4,5,6 (相手4頭) ---
    if (1.0 in {r1, r2} and 2.0 in {r1, r2}) and (3.0 in {r3, r4, r5, r6}):
        hit_7pt += 1

    # --- ② 8点買い: 1,2 - 1,2,3 - 2,3,4,5 ---
    if (1.0 in {r1, r2}) and (2.0 in {r1, r2, r3}) and (3.0 in {r2, r3, r4, r5}) and len({r1,r2,r3}) >= 2:
        hit_8pt += 1

    # --- ③ 10点買い: 1,2 - 1,2,3 - 1,2,3,4,5 (フォーメーション) ---
    if (1.0 in {r1, r2}) and (2.0 in {r1, r2, r3}) and (3.0 in {r1, r2, r3, r4, r5}):
        hit_10pt += 1

    # --- ④ 12点買い: 1位固定 → 2,3,4位 → 2,3,4,5位 (相手4頭マルチ) ---
    if (r1 == 1.0) and (2.0 in {r2, r3, r4}) and (3.0 in {r2, r3, r4, r5}):
        hit_12pt += 1

print("\n" + "="*55)
print("🎯 3連単 7点〜12点 的中率検証結果 (全22,408レース)")
print("="*55)
print(f"🔹 対象レース数 : {total_races:,} レース (全レース勝負)")
print("-" * 55)
print(f"🔥  7点買い (1,2 - 1,2 - 3,4,5,6)  : {hit_7pt / total_races * 100:.1f}%  ({hit_7pt:,}回的中)")
print(f"🔥  8点買い (1,2 - 1,2,3 - 2,3,4,5): {hit_8pt / total_races * 100:.1f}%  ({hit_8pt:,}回的中)")
print(f"🔥 10点買い (1,2 - 1,2,3 - 1~5)    : {hit_10pt / total_races * 100:.1f}%  ({hit_10pt:,}回的中)")
print(f"🔥 12点買い (1固定 → 2,3,4 → 2~5)  : {hit_12pt / total_races * 100:.1f}%  ({hit_12pt:,}回的中)")
print("="*55)