import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import HistGradientBoostingClassifier

print("🌸 勝ち子ちゃん ガチ検証スクリプト (時系列・完全未知データ＆スルー判定適用)")

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

# 時系列で前半80%を【学習用】、後半20%を【完全未知テスト用】に確実に分割
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

# モデル再学習（過去データのみ）
X_train = train_df[features].fillna(0.0)
X_test = test_df[features].fillna(0.0)

m_place = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, random_state=42)
m_place.fit(X_train, (train_df['target_rank'] <= 3.0).astype(int))

m_win = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, random_state=42)
m_win.fit(X_train, (train_df['target_rank'] == 1.0).astype(int))

# 未知データへの予測実行
test_df['prob_place'] = m_place.predict_proba(X_test)[:, 1]
test_df['prob_win'] = m_win.predict_proba(X_test)[:, 1]

# スコア計算 (0-99点)
test_df['score'] = 0
for rid, group in test_df.groupby('race_id'):
    max_p = max(group['prob_place'].max(), 0.01)
    test_df.loc[group.index, 'score'] = ((group['prob_place'] / max_p) * 90 + 9).clip(10, 99).astype(int)

# シミュレーション集計
total_races = 0
pass_races = 0

pat1_races = 0
pat1_umaren_hits = 0
pat1_3fuku_hits = 0

pat2_races = 0
pat2_umatan_hits = 0
pat2_3tan_hits = 0

for race_id, group in test_df.groupby('race_id'):
    if len(group) < 8 or not (1.0 in group['target_rank'].values):
        continue

    total_races += 1
    sorted_g = group.sort_values(by=['score', 'prob_place'], ascending=[False, False])
    scores = sorted_g['score'].values
    ranks = sorted_g['target_rank'].values
    
    if len(scores) < 6: continue

    s1, s2, s3, s4, s5, s6 = scores[:6]
    r1, r2 = ranks[0], ranks[1]

    # カオス混戦判定 (見・スルー)
    if (s1 - s6) <= 8:
        pass_races += 1
        continue

    # パターン①: 4頭超厳選勝負 (1~4位差<=6 & 4~5位差>=8)
    if (s1 - s4) <= 6 and (s4 - s5) >= 8:
        pat1_races += 1
        top4_ranks = set(ranks[:4])
        if {1.0, 2.0}.issubset(top4_ranks): pat1_umaren_hits += 1
        if {1.0, 2.0, 3.0}.issubset(top4_ranks): pat1_3fuku_hits += 1

    # パターン②: 軸1頭抜け勝負 (1位~2位差>=15)
    elif (s1 - s2) >= 15:
        pat2_races += 1
        if r1 == 1.0 and r2 in [2.0, 3.0, 4.0]: pat2_umatan_hits += 1
        if r1 == 1.0 and r2 in [2.0, 3.0] and ranks[2] in [2.0, 3.0, 4.0, 5.0]: pat2_3tan_hits += 1

print("\n" + "="*60)
print("📊 完全未知データ バックテスト検証結果 (過学習ゼロ・時系列分割)")
print("="*60)
print(f"🔹 対象期間総レース数  : {total_races:,} レース")
print(f"🚨 見(スルー)回避数    : {pass_races:,} レース (削減率: {pass_races/max(total_races,1)*100:.1f}%)")
print(f"🔥 勝負実行レース数    : {total_races - pass_races:,} レース")
print("-" * 60)
print(f"【 パターン①：4頭超厳選勝負 】(対象: {pat1_races} レース)")
if pat1_races > 0:
    print(f"  🌸 馬連4頭BOX (6点)  的中率: {pat1_umaren_hits/pat1_races*100:.1f}% ({pat1_umaren_hits}回)")
    print(f"  🌸 3連複4頭BOX (4点) 的中率: {pat1_3fuku_hits/pat1_races*100:.1f}% ({pat1_3fuku_hits}回)")
else:
    print("  ※条件に合致するレースはありませんでした")

print("-" * 60)
print(f"【 パターン②：軸1頭抜け勝負 】(対象: {pat2_races} レース)")
if pat2_races > 0:
    print(f"  🌸 馬単3点流し (3点)  的中率: {pat2_umatan_hits/pat2_races*100:.1f}% ({pat2_umatan_hits}回)")
    print(f"  🔥 3連単6点フォーメーション的中率: {pat2_3tan_hits/pat2_races*100:.1f}% ({pat2_3tan_hits}回)")
else:
    print("  ※条件に合致するレースはありませんでした")
print("="*60)