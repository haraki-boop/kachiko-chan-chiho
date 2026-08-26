import pandas as pd
import numpy as np
import joblib
import os
import re

print("📊 8/25 ガチ検証（的中レース内訳表示版）実行中...")

MODEL_FILE = "keiba_ai_model_nar.pkl"
FUTURE_FILE = "future_races_chiho.csv"
RESULT_FILE = "ml_target_data_chiho.csv"

# 競馬場名を判定するための辞書を追加
NAR_PLACES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

if not os.path.exists(MODEL_FILE) or not os.path.exists(FUTURE_FILE) or not os.path.exists(RESULT_FILE):
    print("⚠️ 必要なファイルが見つかりません。")
    exit()

# 1. 事前データ（future_races_chiho.csv）読み込み
df_future = pd.read_csv(FUTURE_FILE, low_memory=False)

# 2. モデルと特徴量のロード
saved = joblib.load(MODEL_FILE)
model_place = saved['model_place']
model_win = saved['model_win']
features = saved['features']

def get_col(df, cols, default_val):
    for c in cols:
        if c in df.columns:
            return df[c].copy()
    return pd.Series(default_val, index=df.index)

# race_id の文字列型統一（照合エラー防止）
df_future['race_id_clean'] = df_future['race_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

# 前処理（事前データ用）
df_future['first_corner'] = pd.to_numeric(get_col(df_future, ['first_corner', '1角'], 8.0), errors='coerce').fillna(8.0)
df_future['last_corner'] = pd.to_numeric(get_col(df_future, ['last_corner', '4角'], 8.0), errors='coerce').fillna(df_future['first_corner'])
df_future['corner_diff'] = df_future['first_corner'] - df_future['last_corner']

df_future['last_3f'] = pd.to_numeric(get_col(df_future, ['last_3f', '上り'], 39.0), errors='coerce').fillna(39.0)
df_future['time_diff'] = pd.to_numeric(get_col(df_future, ['time_diff', '着差'], 1.5), errors='coerce').fillna(1.5)
df_future['斤量'] = pd.to_numeric(get_col(df_future, ['斤量'], 54.0), errors='coerce').fillna(54.0)
df_future['馬番_num'] = pd.to_numeric(get_col(df_future, ['馬番'], 0), errors='coerce').fillna(0)
df_future['distance_num'] = pd.to_numeric(get_col(df_future, ['distance'], 1400), errors='coerce').fillna(1400)

MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df_future['place_code'] = df_future['race_id_clean'].str[4:6]
df_future['is_minami_kanto'] = df_future['place_code'].isin(MINAMI_KANTO_CODES).astype(int)

# 不足特徴量の補完
for f in features:
    if f not in df_future.columns:
        df_future[f] = 0.0

X_future = df_future[features].fillna(0.0)

# 予想スコア算出（事前データのみで計算）
df_future['p_win'] = model_win.predict_proba(X_future)[:, 1]
df_future['p_rentai'] = model_place.predict_proba(X_future)[:, 1]

df_future['win_norm'] = df_future.groupby('race_id_clean')['p_win'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
df_future['rentai_norm'] = df_future.groupby('race_id_clean')['p_rentai'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
df_future['score'] = df_future['win_norm'] * 0.60 + df_future['rentai_norm'] * 0.30

# 買い目（馬単6点）生成
bets_dict = {}
for race_id, group in df_future.groupby('race_id_clean'):
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

df_result['race_id_clean'] = df_result['race_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
df_result['target_rank'] = df_result['着順'].apply(parse_rank) if '着順' in df_result.columns else df_result.get('target_rank').apply(parse_rank)
df_result['馬番_num'] = pd.to_numeric(get_col(df_result, ['馬番'], 0), errors='coerce').fillna(0)

total_races = 0
hit_races = 0
hit_details = [] # 的中内訳を保存するリストを追加

for race_id, bets in bets_dict.items():
    race_res = df_result[df_result['race_id_clean'] == race_id]
    if race_res.empty: continue
    
    actual_1st = race_res[race_res['target_rank'] == 1.0]['馬番_num'].values
    actual_2nd = race_res[race_res['target_rank'] == 2.0]['馬番_num'].values
    
    if len(actual_1st) > 0 and len(actual_2nd) > 0:
        total_races += 1
        real_top2 = (actual_1st[0], actual_2nd[0])
        if real_top2 in bets:
            hit_races += 1
            # 的中した場合、レース情報をリストに追加
            place_code = race_id[4:6]
            race_num = int(race_id[10:12])
            place_name = NAR_PLACES.get(place_code, "不明")
            hit_details.append(f"{place_name} {race_num}R (馬単: {int(actual_1st[0])}番 → {int(actual_2nd[0])}番)")

hit_rate = (hit_races / total_races * 100) if total_races > 0 else 0.0

print("="*50)
print(f"🔹 対象レース数 : {total_races} レース")
print(f"🎯 的中レース数 : {hit_races} レース")
print(f"📈 リアル的中率 : {hit_rate:.1f} %")
print("-" * 50)
if hit_details:
    print("🎯 的中レース内訳:")
    for detail in hit_details:
        print(f"   ✔️ {detail}")
else:
    print("   的中なし")
print("="*50)