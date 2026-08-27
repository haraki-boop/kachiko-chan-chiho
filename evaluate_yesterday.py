import pandas as pd
import numpy as np
import joblib
import os
import re
import ast

print("📊 第3形態・3連系特化AI ガチ検証（回収率・的中内訳表示版）実行中...")

MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl" # 🌟 第3形態の脳みそに変更
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

# 2. 🌟 第3形態のアンサンブルモデルと特徴量のロード
saved = joblib.load(MODEL_FILE)
features = saved.get('features', [])
m_place_lgb = saved.get('model_place_lgb')
m_win_lgb = saved.get('model_win_lgb')
m_place_xgb = saved.get('model_place_xgb')
m_win_xgb = saved.get('model_win_xgb')
m_place_cat = saved.get('model_place_cat')
m_win_cat = saved.get('model_win_cat')

def get_col(df, cols, default_val):
    for c in cols:
        if c in df.columns:
            return df[c].copy()
    return pd.Series(default_val, index=df.index)

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

X_future = df_future[features].fillna(0.0).astype(float)

# 🌟 アンサンブル予想スコア算出
p_lgb = m_place_lgb.predict_proba(X_future)[:, 1] if m_place_lgb else 0
w_lgb = m_win_lgb.predict_proba(X_future)[:, 1] if m_win_lgb else 0
p_xgb = m_place_xgb.predict_proba(X_future)[:, 1] if m_place_xgb else 0
w_xgb = m_win_xgb.predict_proba(X_future)[:, 1] if m_win_xgb else 0
p_cat = m_place_cat.predict_proba(X_future)[:, 1] if m_place_cat else 0
w_cat = m_win_cat.predict_proba(X_future)[:, 1] if m_win_cat else 0

df_future['p_rentai'] = (p_lgb + p_xgb + p_cat) / 3.0
df_future['p_win'] = (w_lgb + w_xgb + w_cat) / 3.0

df_future['win_norm'] = df_future.groupby('race_id_clean')['p_win'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
df_future['rentai_norm'] = df_future.groupby('race_id_clean')['p_rentai'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
df_future['raw_score'] = df_future['win_norm'] * 0.60 + df_future['rentai_norm'] * 0.30

# 🌟 買い目（3連単・3連複 各6点）の自動生成ロジック
bets_dict = {}
for race_id, group in df_future.groupby('race_id_clean'):
    if len(group) < 5: continue
    
    # スコア100点満点化
    r_max, r_min = group['raw_score'].max(), group['raw_score'].min()
    group['score'] = (((group['raw_score'] - r_min) / (r_max - r_min + 1e-6)) * 100).astype(int)
    
    sorted_group = group.sort_values('score', ascending=False).reset_index(drop=True)
    t1, t2, t3, t4, t5 = sorted_group.loc[0, '馬番_num'], sorted_group.loc[1, '馬番_num'], sorted_group.loc[2, '馬番_num'], sorted_group.loc[3, '馬番_num'], sorted_group.loc[4, '馬番_num']
    
    score_diff = sorted_group.loc[0, 'score'] - sorted_group.loc[1, 'score']
    
    if score_diff >= 5:
        # 1位が抜けている ➔ 3連単（1着:t1, 2・3着: t2,t3,t4 のフォーメーション 計6点）
        bet_type = "3連単"
        bets = [
            (t1, t2, t3), (t1, t2, t4), (t1, t3, t2), (t1, t3, t4), (t1, t4, t2), (t1, t4, t3)
        ]
    else:
        # 大混戦 ➔ 3連複（1軸:t1, 相手: t2,t3,t4,t5 の流し 計6点）
        bet_type = "3連複"
        bets = [
            tuple(sorted((t1, t2, t3))), tuple(sorted((t1, t2, t4))), tuple(sorted((t1, t2, t5))),
            tuple(sorted((t1, t3, t4))), tuple(sorted((t1, t3, t5))), tuple(sorted((t1, t4, t5)))
        ]
        # 重複削除
        bets = list(set(bets))
        
    bets_dict[race_id] = {'type': bet_type, 'bets': bets}

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

# オッズ解析用の関数
def parse_payout(payout_str):
    try:
        # 文字列として入っているリストを実際のリストに変換
        payout_list = ast.literal_eval(payout_str) if isinstance(payout_str, str) else payout_str
        return payout_list
    except:
        return []

total_races = 0
hit_races = 0
total_investment = 0
total_return = 0
hit_details = []

# レース単位での集計（オッズ情報がある場合はそれを取得）
for race_id, strat in bets_dict.items():
    race_res = df_result[df_result['race_id_clean'] == race_id]
    if race_res.empty: continue
    
    # 1〜3着馬を取得
    actual_1st = race_res[race_res['target_rank'] == 1.0]['馬番_num'].values
    actual_2nd = race_res[race_res['target_rank'] == 2.0]['馬番_num'].values
    actual_3rd = race_res[race_res['target_rank'] == 3.0]['馬番_num'].values
    
    if len(actual_1st) > 0 and len(actual_2nd) > 0 and len(actual_3rd) > 0:
        total_races += 1
        # 投資額を加算（1レースにつき6点＝600円）
        total_investment += 600
        
        a1, a2, a3 = actual_1st[0], actual_2nd[0], actual_3rd[0]
        bet_type = strat['type']
        bets = strat['bets']
        
        is_hit = False
        payout = 0
        
        if bet_type == "3連単":
            if (a1, a2, a3) in bets:
                is_hit = True
        elif bet_type == "3連複":
            if tuple(sorted((a1, a2, a3))) in bets:
                is_hit = True
                
        if is_hit:
            hit_races += 1
            # 配当金（オッズ）の取得試行
            payout_col = 'trifecta_payout' if bet_type == "3連単" else 'trio_payout'
            
            # もしCSVに払戻金データがあれば加算、なければ最低保証のダミー値(1000円)を入れる
            if payout_col in race_res.columns:
                payout_data = parse_payout(race_res[payout_col].iloc[0])
                if payout_data and len(payout_data) > 0:
                    try:
                        payout = int(str(payout_data[0]).replace(',', ''))
                    except:
                        payout = 1000
                else:
                    payout = 1000
            else:
                payout = 1000
            
            total_return += payout
            
            place_code = race_id[4:6]
            race_num = int(race_id[10:12])
            place_name = NAR_PLACES.get(place_code, "地方")
            
            hit_details.append(f"🎯 {place_name} {race_num}R [{bet_type}] 的中！ 払戻: {payout:,}円")

hit_rate = (hit_races / total_races * 100) if total_races > 0 else 0.0
return_rate = (total_return / total_investment * 100) if total_investment > 0 else 0.0
profit = total_return - total_investment

print("="*50)
print(f"🔹 対象レース数 : {total_races} レース")
print(f"🎯 的中レース数 : {hit_races} レース")
print(f"📈 リアル的中率 : {hit_rate:.1f} %")
print("-" * 50)
print(f"💰 総投資額   : {total_investment:,} 円")
print(f"💸 総払戻額   : {total_return:,} 円")
print(f"📊 回収率     : {return_rate:.1f} %")
print(f"💵 収支       : {'+' if profit >= 0 else ''}{profit:,} 円")
print("-" * 50)
if hit_details:
    print("🎯 的中レース内訳:")
    for detail in hit_details:
        print(f"   {detail}")
else:
    print("   的中なし")
print("="*50)