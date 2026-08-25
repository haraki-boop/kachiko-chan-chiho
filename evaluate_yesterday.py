import pandas as pd
import numpy as np
import joblib
import os
import re

print("📊 8/25 ガチ検証（完全リーク遮断版）実行中...")

MODEL_FILE = "keiba_ai_model_nar.pkl"
FUTURE_FILE = "future_races_chiho.csv"
RESULT_FILE = "ml_target_data_chiho.csv"

if not os.path.exists(MODEL_FILE) or not os.path.exists(FUTURE_FILE) or not os.path.exists(RESULT_FILE):
    print("⚠️ 必要なファイルが見つかりません。")
    exit()

# 1. 事前データ（future_races_chiho.csv）の読み込み
df_future = pd.read_csv(FUTURE_FILE, low_memory=False)

# 2. モデルと特徴量のロード
saved = joblib.load(MODEL_FILE)
model_place = saved['model_place']
model_win = saved['model_win']
features = saved['features']

# 前処理（事前データ用）
df_future['first_corner'] = pd.to_numeric(df_future.get('first_corner', df_future.get('1角')), errors='coerce').fillna(8.0)
df_future['last_corner'] = pd.to_numeric(df_future.get('last_corner', df_future.get('4角')), errors='coerce').fillna(df_future['first_corner'])
df_future['corner_diff'] = df_future['first_corner'] - df_future['last_corner']
df_future['last_3f'] = pd.to_numeric(df_future.get('last_3f', df_future.get('上り')), errors='coerce')
df_future['time_diff'] = pd.to_numeric(df_future.get('time_diff', df_future.get('着差')), errors='coerce').fillna(1.5)
df_future['斤量'] = pd.to_numeric(df_future.get('斤量'), errors='coerce').fillna(54.0)
df_future['馬番_num'] = pd.to_numeric(df_future.get('馬番'), errors='coerce').fillna(0)
df_future['distance_num'] = pd.to_numeric(df_future.get('distance'), errors='coerce').fillna(1400)

MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df_future['place_code'] = df_future['race_id'].astype(str).str[4:6]
df_future['is_minami_kanto'] = df_future['place_code'].isin(MINAMI_KANTO_CODES).astype(int)

# 不足している特徴量はデフォルト値で補完（完全クリーンな事前予測）
for f in features:
    if f not in df_future.columns:
        df_future[f] = 0.0

X_future = df_future[features].fillna(0.0)

# 予想スコア算出（着順未知のデータのみ使用）
df_future['p_win'] = model_win.predict_proba(X_future)[:, 1]
df_future['p_rentai'] = model_place.predict_proba(X_future)[:, 1]

df_future['win_norm'] = df_future.groupby('race_id')['p_win'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
df_future['rentai_norm'] = df_future.groupby('race_id')['p_rentai'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
df_future['score'] = df_future['win_norm'] * 0.60 + df_future['rentai_norm'] * 0.30

# 買い目（馬単6点）生成
bets_dict = {}
for race_id, group in df_future.groupby('race_id'):
    if len(group) < 4: continue
    sorted_group = group.sort_values('score', ascending=False).reset_index(drop=True)
    t1, t2, t3, t4 = sorted_group.loc[0, '馬番_num'], sorted_group.loc[1, '馬番_num'], sorted_group.loc[2, '馬番_num'], sorted_group.loc[3, '馬番_num']
    bets_dict[race_id] = [(t1, t2), (t1, t3), (t1, t4), (t2, t1), (t2, t3), (t2, t4)]

# 3. 確定結果（ml_target_data_chiho.csv）と照合
df_result = pd.read_csv(RESULT_FILE, low_memory=False)

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

df_result['target_rank'] = df_result['着順'].apply(parse_rank) if '着順' in df_result.columns else df_result.get('target_rank').apply(parse_rank)
df_result['馬番_num'] = pd.to_numeric(df_result.get('馬番'), errors='coerce').fillna(0)

total_races = 0
hit_races = 0

for race_id, bets in bets_dict.items():
    race_res = df_result[df_result['race_id'] == race_id]
    if race_res.empty: continue
    
    actual_1st = race_res[race_res['target_rank'] == 1.0]['馬番_num'].values
    actual_2nd = race_res[race_res['target_rank'] == 2.0]['馬番_num'].values
    
    if len(actual_1st) > 0 and len(actual_2nd) > 0:
        total_races += 1
        real_top2 = (actual_1st[0], actual_2nd[0])
        if real_top2 in bets:
            hit_races += 1

hit_rate = (hit_races / total_races * 100) if total_races > 0 else 0.0

print("="*50)
print(f"🔹 対象レース数 : {total_races} レース")
print(f"🎯 的中レース数 : {hit_races} レース")
print(f"📈 リアル的中率 : {hit_rate:.1f} %")
print("="*50)