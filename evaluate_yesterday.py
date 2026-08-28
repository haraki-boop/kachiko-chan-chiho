import pandas as pd
import numpy as np
import joblib
import os
import re
import ast
import unicodedata

print("📊 第3形態・3連系特化AI ガチ検証（回収率・的中内訳表示版）実行中...")

MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"
FUTURE_FILE = "future_races_chiho.csv"
RESULT_FILE = "ml_target_data_chiho.csv"

NAR_PLACES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

if not os.path.exists(MODEL_FILE) or not os.path.exists(FUTURE_FILE) or not os.path.exists(RESULT_FILE):
    print("⚠️ 必要なファイルが見つかりません。")
    exit()

def clean_horse_name(name): 
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･._ ]+', '', s).strip()

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

# 1. 過去データの読み込み & 辞書構築
try:
    df_result = pd.read_csv(RESULT_FILE, low_memory=False, encoding='utf-8')
except UnicodeDecodeError:
    df_result = pd.read_csv(RESULT_FILE, low_memory=False, encoding='cp932')

try:
    df_future = pd.read_csv(FUTURE_FILE, low_memory=False, encoding='utf-8')
except UnicodeDecodeError:
    df_future = pd.read_csv(FUTURE_FILE, low_memory=False, encoding='cp932')

df_result['馬名_clean'] = df_result['馬名'].astype(str).apply(clean_horse_name)
df_result['騎手_clean'] = df_result.get('騎手', pd.Series(['']*len(df_result))).astype(str).apply(clean_horse_name)
trainer_col = df_result.get('調教師', df_result['騎手_clean'])
df_result['trainer_clean'] = trainer_col.astype(str).apply(clean_horse_name)
df_result['jockey_trainer_combo'] = df_result['騎手_clean'] + "_" + df_result['trainer_clean']

rank_col = '着順_num' if '着順_num' in df_result.columns else '着順'
df_result['target_rank_tmp'] = df_result[rank_col].apply(parse_rank)
df_result['target_win'] = (df_result['target_rank_tmp'] == 1.0).astype(int)

df_result['first_corner'] = pd.to_numeric(df_result.get('first_corner', df_result.get('1角')), errors='coerce').fillna(8.0)
df_result['last_corner'] = pd.to_numeric(df_result.get('last_corner', df_result.get('4角')), errors='coerce').fillna(df_result['first_corner'])
df_result['last_3f'] = pd.to_numeric(df_result.get('last_3f', df_result.get('上り')), errors='coerce').fillna(39.0)
df_result['time_diff'] = pd.to_numeric(df_result.get('time_diff', df_result.get('着差')), errors='coerce').fillna(1.5)
df_result['distance_num'] = pd.to_numeric(df_result.get('distance'), errors='coerce').fillna(1400)
df_result['place_code_tmp'] = df_result['race_id'].astype(str).str[4:6]

# 🌟 指数データと賞金データの前処理（追加）
df_result['custom_time_index'] = pd.to_numeric(df_result.get('custom_time_index'), errors='coerce').fillna(100.0)
df_result['custom_start_index'] = pd.to_numeric(df_result.get('custom_start_index'), errors='coerce').fillna(50.0)
df_result['custom_last3f_index'] = pd.to_numeric(df_result.get('custom_last3f_index'), errors='coerce').fillna(50.0)
df_result['prize_num'] = pd.to_numeric(df_result.get('賞金(万円)', 0), errors='coerce').fillna(0.0)
df_result['prize_num_log'] = np.log1p(df_result['prize_num'])

MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df_result['is_minami_kanto'] = df_result['place_code_tmp'].isin(MINAMI_KANTO_CODES).astype(int)

jockey_dict = df_result.groupby('騎手_clean')['target_win'].mean().to_dict()
trainer_dict = df_result.groupby('trainer_clean')['target_win'].mean().to_dict()
combo_dict = df_result.groupby('jockey_trainer_combo')['target_win'].mean().to_dict()

df_result['waku_num_tmp'] = pd.to_numeric(df_result.get('枠番'), errors='coerce').fillna(0)
df_result['place_waku_combo'] = df_result['place_code_tmp'] + "_" + df_result['waku_num_tmp'].astype(str)
waku_dict = df_result.groupby('place_waku_combo')['target_win'].mean().to_dict()

df_result['date_dt'] = pd.to_datetime(df_result.get('date'), errors='coerce').fillna(pd.to_datetime('2020-01-01'))

baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df_result['baba_code'] = df_result.get('馬場', pd.Series(['良']*len(df_result))).map(baba_map).fillna(1)
df_result['is_bad_baba'] = (df_result['baba_code'] >= 3).astype(int)

horse_dict = {}
for h, group in df_result.sort_values('date_dt').groupby('馬名_clean'):
    r3 = group.tail(3)
    r5 = group.tail(5)
    f_c = r3['first_corner'].mean()
    l_c = r3['last_corner'].mean()
    l_3f = r3['last_3f'].mean()
    t_diff = r3['time_diff'].mean()
    avg_rank_3 = r3['target_rank_tmp'].mean()
    avg_rank_5 = r5['target_rank_tmp'].mean()
    dist_dict = group.groupby('distance_num')['target_rank_tmp'].apply(lambda x: x.tail(3).mean()).to_dict()
    place_dict = group.groupby('place_code_tmp')['target_rank_tmp'].apply(lambda x: x.tail(3).mean()).to_dict()
    
    # 🌟 指数・賞金の平均計算（追加）
    time_idx_avg = r3['custom_time_index'].mean()
    start_idx_avg = r3['custom_start_index'].mean()
    last3f_idx_avg = r3['custom_last3f_index'].mean()
    horse_prize_avg = r5['prize_num_log'].mean()

    bad_baba_mean = group[group['is_bad_baba'] == 1]['target_rank_tmp'].tail(3).mean()
    if pd.isna(bad_baba_mean): bad_baba_mean = avg_rank_3

    last_row = group.iloc[-1]
    prev_is_minami = last_row.get('is_minami_kanto', 0)
    last_date = group['date_dt'].max()
    days_since = (pd.Timestamp.now() - last_date).days if not pd.isna(last_date) else 14.0
    
    horse_dict[h] = {
        'first_corner': f_c, 'last_corner': l_c, 'corner_diff': f_c - l_c,
        'last_3f': l_3f, 'time_diff': t_diff, 'recent_avg_rank_3': avg_rank_3,
        'recent_avg_rank_5': avg_rank_5, 'days_since_prev': days_since,
        'horse_career_runs': len(group), 'prev_is_minami': prev_is_minami,
        'dist_dict': dist_dict, 'place_dict': place_dict, 'bad_baba_avg_rank': bad_baba_mean,
        'prev_time_index_avg': time_idx_avg, 'prev_start_index_avg': start_idx_avg, 
        'prev_last3f_index_avg': last3f_idx_avg, 'horse_prize_avg': horse_prize_avg # 🌟 保存
    }

# 2. モデルロード
saved = joblib.load(MODEL_FILE)
features = saved.get('features', [])
# 🌟 LambdaMART Rankerモデルの取得
m_lgb = saved.get('model_rank_lgb')
m_xgb = saved.get('model_rank_xgb')
m_cat = saved.get('model_rank_cat')

if not m_lgb and not m_xgb and not m_cat:
    print("⚠️ エラー: Rankingモデルが見つかりません。")
    exit()

# 3. 出馬表（df_future）へ特徴量を正確に結合
df_future['race_id_clean'] = df_future['race_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
df_future['馬名_clean'] = df_future['馬名'].astype(str).apply(clean_horse_name)
df_future['騎手_clean'] = df_future.get('騎手', pd.Series(['']*len(df_future))).astype(str).apply(clean_horse_name)
trainer_col = df_future.get('調教師', df_future['騎手_clean'])
df_future['trainer_clean'] = trainer_col.astype(str).apply(clean_horse_name)
df_future['jockey_trainer_combo'] = df_future['騎手_clean'] + "_" + df_future['trainer_clean']

df_future['place_code'] = df_future['race_id_clean'].str[4:6]
df_future['is_minami_kanto'] = df_future['place_code'].isin(MINAMI_KANTO_CODES).astype(int)
df_future['prev_is_minami'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_is_minami', 0))
df_future['recent_avg_rank_3'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank_3', 5.0))
df_future['recent_avg_rank_5'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank_5', 5.0))
df_future['distance_num'] = pd.to_numeric(df_future.get('distance'), errors='coerce').fillna(1400)

def get_same_dist(row):
    h, d = row['馬名_clean'], row['distance_num']
    return horse_dict.get(h, {}).get('dist_dict', {}).get(d, horse_dict.get(h, {}).get('recent_avg_rank_3', 5.0))

def get_same_place(row):
    h, p = row['馬名_clean'], str(row['place_code']).zfill(2)
    return horse_dict.get(h, {}).get('place_dict', {}).get(p, horse_dict.get(h, {}).get('recent_avg_rank_3', 5.0))

df_future['same_dist_avg_rank'] = df_future.apply(get_same_dist, axis=1)
df_future['same_place_avg_rank'] = df_future.apply(get_same_place, axis=1)
df_future['days_since_prev'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('days_since_prev', 14.0))

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0) if m else (470.0, 0.0)

if '馬体重' in df_future.columns:
    parsed = df_future['馬体重'].apply(parse_weight_info)
    df_future['body_weight'] = parsed.apply(lambda x: x[0])
    df_future['body_weight_diff'] = parsed.apply(lambda x: x[1])
else: 
    df_future['body_weight'] = 470.0
    df_future['body_weight_diff'] = 0.0

df_future['is_large_weight_change'] = (df_future['body_weight_diff'].abs() >= 10.0).astype(int)

df_future['prev_1c'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('first_corner', 8.0))
df_future['last_corner'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('last_corner', 8.0))
df_future['corner_diff'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('corner_diff', 0.0))
df_future['last_3f_avg_rank'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('last_3f', 39.0))
df_future['avg_time_diff'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('time_diff', 1.5))
df_future['bad_baba_avg_rank'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('bad_baba_avg_rank', 5.0))
df_future['is_bad_baba'] = 0
df_future['horse_career_runs'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('horse_career_runs', 5.0))
df_future['斤量'] = pd.to_numeric(df_future.get('斤量'), errors='coerce').fillna(54.0)
df_future['kinryo_weight_ratio'] = df_future['斤量'] / df_future['body_weight'].clip(lower=350.0)

df_future['is_front_runner'] = (df_future['prev_1c'] <= 3.0).astype(int)
front_runners = df_future.groupby('race_id_clean')['is_front_runner'].transform('sum')
df_future['race_front_runners'] = front_runners

df_future['waku_num'] = pd.to_numeric(df_future.get('枠番'), errors='coerce').fillna(0)
df_future['place_waku_combo'] = df_future['place_code'] + "_" + df_future['waku_num'].astype(str)
df_future['waku_win_rate'] = df_future['place_waku_combo'].apply(lambda x: waku_dict.get(x, 0.05))
df_future['jockey_win_rate'] = df_future['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
df_future['trainer_win_rate'] = df_future['trainer_clean'].apply(lambda x: trainer_dict.get(x, 0.05))
df_future['combo_win_rate'] = df_future['jockey_trainer_combo'].apply(lambda x: combo_dict.get(x, 0.05))

# 🌟 不足していた特徴量の呼び出し
df_future['horse_prize_avg'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('horse_prize_avg', 0.0))
race_mean_prize = df_future.groupby('race_id_clean')['horse_prize_avg'].transform('mean').clip(lower=0.1)
df_future['race_prize_relative'] = df_future['horse_prize_avg'] / race_mean_prize
df_future['race_prize_rank'] = df_future.groupby('race_id_clean')['horse_prize_avg'].rank(ascending=False, method='min')

df_future['prev_time_index_avg'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_time_index_avg', 100.0))
df_future['prev_start_index_avg'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_start_index_avg', 50.0))
df_future['prev_last3f_index_avg'] = df_future['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_last3f_index_avg', 50.0))
df_future['dist_change_num'] = pd.to_numeric(df_future.get('dist_change', pd.Series([0.0]*len(df_future))), errors='coerce').fillna(0.0)

df_future['馬番_num'] = pd.to_numeric(df_future.get('馬番'), errors='coerce').fillna(0)

# 4. 推論実行
X_future = df_future[features].fillna(0.0).astype(float)

# 🌟 Rankerモデルの予測（predictを使用）
preds = []
if m_lgb and hasattr(m_lgb, 'predict'): preds.append(m_lgb.predict(X_future))
if m_xgb and hasattr(m_xgb, 'predict'): preds.append(m_xgb.predict(X_future))
if m_cat and hasattr(m_cat, 'predict'): preds.append(m_cat.predict(X_future))

if preds:
    df_future['raw_score'] = np.mean(preds, axis=0)
else:
    print("⚠️ 予測に失敗しました。")
    exit()

# 5. 買い目生成 & 答え合わせ
bets_dict = {}
for race_id, group in df_future.groupby('race_id_clean'):
    if len(group) < 5: continue
    
    r_max, r_min = group['raw_score'].max(), group['raw_score'].min()
    group['score'] = (((group['raw_score'] - r_min) / (r_max - r_min + 1e-6)) * 100).astype(int)
    
    sorted_group = group.sort_values('score', ascending=False).reset_index(drop=True)
    t1, t2, t3, t4, t5 = sorted_group.loc[0, '馬番_num'], sorted_group.loc[1, '馬番_num'], sorted_group.loc[2, '馬番_num'], sorted_group.loc[3, '馬番_num'], sorted_group.loc[4, '馬番_num']
    score_diff = sorted_group.loc[0, 'score'] - sorted_group.loc[1, 'score']
    
    if score_diff >= 5:
        bet_type = "3連単"
        bets = [(t1, t2, t3), (t1, t2, t4), (t1, t3, t2), (t1, t3, t4), (t1, t4, t2), (t1, t4, t3)]
    else:
        bet_type = "3連複"
        bets = list(set([
            tuple(sorted((t1, t2, t3))), tuple(sorted((t1, t2, t4))), tuple(sorted((t1, t2, t5))),
            tuple(sorted((t1, t3, t4))), tuple(sorted((t1, t3, t5))), tuple(sorted((t1, t4, t5)))
        ]))
        
    bets_dict[race_id] = {'type': bet_type, 'bets': bets}

df_result['race_id_clean'] = df_result['race_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
df_result['target_rank'] = df_result['target_rank_tmp']
df_result['馬番_num'] = pd.to_numeric(df_result.get('馬番'), errors='coerce').fillna(0)

def parse_payout(payout_str):
    try:
        payout_list = ast.literal_eval(payout_str) if isinstance(payout_str, str) else payout_str
        return payout_list
    except:
        return []

total_races = 0
hit_races = 0
total_investment = 0
total_return = 0
hit_details = []

for race_id, strat in bets_dict.items():
    race_res = df_result[df_result['race_id_clean'] == race_id]
    if race_res.empty: continue
    
    actual_1st = race_res[race_res['target_rank'] == 1.0]['馬番_num'].values
    actual_2nd = race_res[race_res['target_rank'] == 2.0]['馬番_num'].values
    actual_3rd = race_res[race_res['target_rank'] == 3.0]['馬番_num'].values
    
    if len(actual_1st) > 0 and len(actual_2nd) > 0 and len(actual_3rd) > 0:
        total_races += 1
        total_investment += 600
        
        a1, a2, a3 = actual_1st[0], actual_2nd[0], actual_3rd[0]
        bet_type = strat['type']
        bets = strat['bets']
        
        is_hit = False
        payout = 0
        
        if bet_type == "3連単" and (a1, a2, a3) in bets:
            is_hit = True
        elif bet_type == "3連複" and tuple(sorted((a1, a2, a3))) in bets:
            is_hit = True
                
        if is_hit:
            hit_races += 1
            # "3連単"の払い戻しカラム名に合わせて修正
            payout_col = 'trifecta_payout' if bet_type == "3連単" else 'trio_payout'
            
            if payout_col in race_res.columns:
                payout_data = parse_payout(race_res[payout_col].iloc[0])
                if payout_data and len(payout_data) > 0:
                    try: payout = int(str(payout_data[0]).replace(',', ''))
                    except: payout = 1000
                else: payout = 1000
            else: payout = 1000
            
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