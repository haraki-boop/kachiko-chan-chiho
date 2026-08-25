import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import HistGradientBoostingClassifier

print("🌸 勝ち子ちゃん ガチ検証スクリプト (純粋確率・馬単フォーメーション6点 特化版)")

CSV_FILE = "ml_target_data_chiho.csv"

if not os.path.exists(CSV_FILE):
    print("⚠️ ml_target_data_chiho.csv が見つかりません。")
    exit()

print("📊 データを読み込み、日付・時系列処理を実行中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

df['target_rank'] = df['着順'].apply(parse_rank).fillna(6.0) if '着順' in df.columns else df.get('target_rank').apply(parse_rank).fillna(6.0)
df = df[df['target_rank'] < 90.0].copy()

# ターゲット設定（連対：2着以内 と 勝利：1着）
df['target_rentai'] = (df['target_rank'] <= 2.0).astype(int)
df['target_win'] = (df['target_rank'] == 1.0).astype(int)

# 配当計算用：単勝オッズの取得
df['単勝_num'] = pd.to_numeric(df.get('単勝'), errors='coerce').fillna(15.0)

# 日付順に厳格ソート
if 'date' in df.columns:
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['date_dt', 'race_id']).reset_index(drop=True)
else:
    df = df.sort_values(['race_id']).reset_index(drop=True)

# 1. 特徴量パース
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

MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df['place_code'] = df['race_id'].astype(str).str[4:6]
df['is_minami_kanto'] = df['place_code'].isin(MINAMI_KANTO_CODES).astype(int)

baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df['baba_code'] = df.get('馬場', pd.Series(['良']*len(df))).map(baba_map).fillna(1)
df['is_bad_baba'] = (df['baba_code'] >= 3).astype(int)

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

# 未来リークを防止する過去走集計
df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['recent_avg_rank_5'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['last_3f_avg_rank'] = df.groupby('馬名_clean')['last_3f'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(39.0))
df['avg_time_diff'] = df.groupby('馬名_clean')['time_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(1.5))

df['bad_baba_avg_rank'] = df[df['is_bad_baba'] == 1].groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['bad_baba_avg_rank'] = df.groupby('馬名_clean')['bad_baba_avg_rank'].ffill().fillna(df['recent_avg_rank_3'])

df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['same_place_avg_rank'] = df.groupby(['馬名_clean', 'place_code'])['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())

df['days_since_prev'] = 30.0
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()
df['prev_is_minami'] = df.groupby('馬名_clean')['is_minami_kanto'].shift().fillna(0).astype(int)

df['custom_time_index'] = 75.0 - (df['recent_avg_rank_3'].fillna(6.0).clip(1, 14) - 3.0) * 3.5 + (df['斤量'] - 54.0) * 1.5
df['custom_start_index'] = (12.0 - df['prev_1c'].fillna(8.0).clip(upper=10.0)) * 6.5

trainer_col = '調教師' if '調教師' in df.columns else '騎手'
df['trainer_clean'] = df[trainer_col].astype(str)
df['owner_clean'] = df['馬主'].astype(str) if '馬主' in df.columns else ''
df['jockey_trainer_combo'] = df['騎手'].astype(str) + "_" + df['trainer_clean']

features = [
    'is_minami_kanto', 'prev_is_minami', 'recent_avg_rank_3', 'recent_avg_rank_5', 
    'same_dist_avg_rank', 'same_place_avg_rank', 'days_since_prev', 'is_large_weight_change',
    'prev_1c', 'last_corner', 'corner_diff', 'last_3f_avg_rank', 'avg_time_diff', 'bad_baba_avg_rank', 'is_bad_baba',
    'horse_career_runs', 'custom_time_index', 'custom_start_index', 
    '斤量', 'sex_code', 'age', 'body_weight', 'body_weight_diff', 
    'kinryo_weight_ratio', 'distance_num', '馬番_num'
]

# 時系列で前半80%を【学習用】、後半20%を【完全未知テスト用】に分割
split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"🧠 学習期間データ数 (過去80%): {len(train_df):,} 行")
print(f"🎯 検証期間データ数 (未知20%): {len(test_df):,} 行")

# 学習データのみで騎手・調教師の勝率を過去計算
trainer_stats = (train_df[train_df['target_rank'] == 1.0].groupby('trainer_clean')['target_rank'].count() / train_df.groupby('trainer_clean')['target_rank'].count()).to_dict()
combo_stats = (train_df[train_df['target_rank'] == 1.0].groupby('jockey_trainer_combo')['target_rank'].count() / train_df.groupby('jockey_trainer_combo')['target_rank'].count()).to_dict()
jockey_stats = (train_df[train_df['target_rank'] == 1.0].groupby('騎手')['target_rank'].count() / train_df.groupby('騎手')['target_rank'].count()).to_dict()

for t_data in [train_df, test_df]:
    t_data['trainer_win_rate'] = t_data['trainer_clean'].map(trainer_stats).fillna(0.05)
    t_data['combo_win_rate'] = t_data['jockey_trainer_combo'].map(combo_stats).fillna(0.05)
    t_data['jockey_win_rate'] = t_data['騎手'].map(jockey_stats).fillna(0.05)

features.extend(['trainer_win_rate', 'combo_win_rate', 'jockey_win_rate'])

X_train = train_df[features].fillna(0.0)
X_test = test_df[features].fillna(0.0)

y_train_rentai = train_df['target_rentai']
y_train_win = train_df['target_win']

# 🚨 ペナルティ(sample_weight)を排除した純粋確率学習
m_place = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, random_state=42)
m_place.fit(X_train, y_train_rentai)

m_win = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, random_state=42)
m_win.fit(X_train, y_train_win)

# 未知データへの予測実行
test_df['prob_place'] = m_place.predict_proba(X_test)[:, 1] # 実質はprob_rentai
test_df['prob_win'] = m_win.predict_proba(X_test)[:, 1]

# アプリ本番と同じハイブリッドスコア計算
test_df['score_brain'] = 0
for rid, group in test_df.groupby('race_id'):
    w_norm = (group['prob_win'] - group['prob_win'].min()) / (group['prob_win'].max() - group['prob_win'].min() + 1e-6)
    p_norm = (group['prob_place'] - group['prob_place'].min()) / (group['prob_place'].max() - group['prob_place'].min() + 1e-6)
    front_bonus = np.where(group['first_corner'] <= 3.0, 0.15, 0.0)
    
    raw_score = (w_norm * 0.60) + (p_norm * 0.30) + front_bonus
    test_df.loc[group.index, 'score_brain'] = (((raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-6)) * 89 + 10).astype(int)

# --------------------------------------------------------
# 💰 馬単フォーメーション6点 シミュレーション
# --------------------------------------------------------
total_races = 0
hit_races = 0
total_investment = 0
total_return = 0

for race_id, group in test_df.groupby('race_id'):
    if len(group) < 8 or not (1.0 in group['target_rank'].values) or not (2.0 in group['target_rank'].values):
        continue

    total_races += 1
    # スコア順 ＞ 1着率順 でソート
    sorted_g = group.sort_values(by=['score_brain', 'prob_win'], ascending=[False, False])
    
    if len(sorted_g) < 4: continue
    
    top4_horses = sorted_g.iloc[:4]
    horse_1st_pred = top4_horses.iloc[0]['馬番_num']
    horse_2nd_pred = top4_horses.iloc[1]['馬番_num']
    horse_3rd_pred = top4_horses.iloc[2]['馬番_num']
    horse_4th_pred = top4_horses.iloc[3]['馬番_num']

    # 実際の1着馬と2着馬
    actual_1st = group[group['target_rank'] == 1.0].iloc[0]['馬番_num']
    actual_2nd = group[group['target_rank'] == 2.0].iloc[0]['馬番_num']

    # 馬単フォーメーション (1・2位 ➔ 1〜4位) の買い目 (6点)
    buy_patterns = [
        (horse_1st_pred, horse_2nd_pred), (horse_1st_pred, horse_3rd_pred), (horse_1st_pred, horse_4th_pred),
        (horse_2nd_pred, horse_1st_pred), (horse_2nd_pred, horse_3rd_pred), (horse_2nd_pred, horse_4th_pred)
    ]
    
    total_investment += 600 # 1点100円 × 6点
    
    # 的中判定
    is_hit = False
    for buy_1st, buy_2nd in buy_patterns:
        if actual_1st == buy_1st and actual_2nd == buy_2nd:
            is_hit = True
            hit_races += 1
            
            # 【仮配当計算】
            odds_1st = group[group['馬番_num'] == actual_1st]['単勝_num'].values[0]
            odds_2nd = group[group['馬番_num'] == actual_2nd]['単勝_num'].values[0]
            
            simulated_payout = max((odds_1st * odds_2nd * 100 * 0.8), 300)
            total_return += simulated_payout
            break

print("\n" + "="*60)
print("📊 黄金・馬単フォーメーション(6点) バックテスト結果")
print("="*60)
print(f"🔹 対象レース数 : {total_races:,} レース")
print(f"🎯 的中レース数 : {hit_races:,} レース")
print(f"📈 的中率       : {(hit_races/total_races)*100:.1f} %")
print("-" * 60)
print(f"💸 総投資額     : {total_investment:,.0f} 円 (全レース 600円ベタ買い)")
print(f"💰 仮想総回収額 : {total_return:,.0f} 円 (※単勝オッズからの推定値)")
print(f"📊 仮想回収率   : {(total_return/total_investment)*100:.1f} %")
print("="*60)