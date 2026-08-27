import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import os
import re

print("🚀 勝ち子ちゃん 【LambdaMART 順位学習(Ranking) × クラス・格付け特化版】")

CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"

if not os.path.exists(CSV_FILE):
    print(f"⚠️ {CSV_FILE} が見つかりません。")
    exit()

print("📊 過去データを読み込み、前処理および格付け・メンバーレベル特徴量を生成中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

def parse_rank(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return np.nan

target_col = '着順_num' if '着順_num' in df.columns else '着順'
df['target_rank_clean'] = df[target_col].apply(parse_rank)
df = df[df['target_rank_clean'].notna() & (df['target_rank_clean'] < 90.0)].copy()

# レース内順位の関連度（Relevance）スコア作成（1着=5, 2着=4, 3着=3, 4着=2, 5着=1, 6着以下=0）
def rank_to_relevance(rank):
    if rank == 1.0: return 5
    elif rank == 2.0: return 4
    elif rank == 3.0: return 3
    elif rank == 4.0: return 2
    elif rank == 5.0: return 1
    else: return 0

df['relevance'] = df['target_rank_clean'].apply(rank_to_relevance)

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

# 🌟 【新機能】格・賞金・メンバーレベル比較特徴量の生成
df['prize_num'] = pd.to_numeric(df.get('賞金(万円)', 0), errors='coerce').fillna(0.0)
df['horse_prize_avg'] = df.groupby('馬名_clean')['prize_num'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean().fillna(0.0))

# 時系列ソート & レースID順に並べ替え（グループ化の必須条件）
df = df.sort_values(['date', 'race_id']).reset_index(drop=True)

# レース内における各馬の「相対的な格（平均獲得賞金比率）」と「メンバー内格付け順位」
df['race_prize_mean'] = df.groupby('race_id')['horse_prize_avg'].transform('mean').clip(lower=0.1)
df['race_prize_relative'] = df['horse_prize_avg'] / df['race_prize_mean']
df['race_prize_rank'] = df.groupby('race_id')['horse_prize_avg'].rank(ascending=False, method='min')

MINAMI_KANTO_CODES = ['42', '43', '44', '45']
df['place_code'] = df['race_id'].astype(str).str[4:6]
df['is_minami_kanto'] = df['place_code'].isin(MINAMI_KANTO_CODES).astype(int)

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0) if m else (470.0, 0.0)

if '馬体重' in df.columns:
    parsed = df['馬体重'].apply(parse_weight_info)
    df['body_weight'] = parsed.apply(lambda x: x[0])
    df['body_weight_diff'] = parsed.apply(lambda x: x[1])
else: 
    df['body_weight'] = 470.0
    df['body_weight_diff'] = 0.0

df['kinryo_weight_ratio'] = df['斤量'] / df['body_weight'].clip(lower=350.0)
df['is_large_weight_change'] = (df['body_weight_diff'].abs() >= 10.0).astype(int)

baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
df['baba_code'] = df.get('馬場', pd.Series(['良']*len(df))).map(baba_map).fillna(1)
df['is_bad_baba'] = (df['baba_code'] >= 3).astype(int)

# 成績シフト集計
df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(5.0))
df['recent_avg_rank_5'] = df.groupby('馬名_clean')['target_rank_clean'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean().fillna(5.0))
df['prev_1c'] = df.groupby('馬名_clean')['first_corner'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(8.0))
df['last_3f_avg_rank'] = df.groupby('馬名_clean')['last_3f'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(39.0))
df['avg_time_diff'] = df.groupby('馬名_clean')['time_diff'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(1.5))
df['bad_baba_avg_rank'] = df[df['is_bad_baba'] == 1].groupby('馬名_clean')['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
df['bad_baba_avg_rank'] = df.groupby('馬名_clean')['bad_baba_avg_rank'].ffill().fillna(df['recent_avg_rank_3'])
df['same_dist_avg_rank'] = df.groupby(['馬名_clean', 'distance_num'])['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(5.0))
df['same_place_avg_rank'] = df.groupby(['馬名_clean', 'place_code'])['target_rank_clean'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean().fillna(5.0))
df['prev_date'] = df.groupby('馬名_clean')['date'].shift()
df['days_since_prev'] = (df['date'] - df['prev_date']).dt.days.fillna(14.0)
df['horse_career_runs'] = df.groupby('馬名_clean').cumcount()
df['prev_is_minami'] = df.groupby('馬名_clean')['is_minami_kanto'].shift().fillna(0).astype(int)

df['is_front_runner'] = (df['prev_1c'] <= 3.0).astype(int)
df['race_front_runners'] = df.groupby('race_id')['is_front_runner'].transform('sum')

df['place_waku_combo'] = df['place_code'].astype(str) + "_" + df['waku_num'].astype(str)
waku_stats = (df[df['target_rank_clean'] == 1.0].groupby('place_waku_combo')['target_rank_clean'].count() / df.groupby('place_waku_combo')['target_rank_clean'].count()).to_dict()
df['waku_win_rate'] = df['place_waku_combo'].map(waku_stats).fillna(0.05)

trainer_col = '調教師' if '調教師' in df.columns else '騎手'
df['trainer_clean'] = df[trainer_col].astype(str)
df['jockey_trainer_combo'] = df['騎手'].astype(str) + "_" + df['trainer_clean']
trainer_stats = (df[df['target_rank_clean'] == 1.0].groupby('trainer_clean')['target_rank_clean'].count() / df.groupby('trainer_clean')['target_rank_clean'].count()).to_dict()
combo_stats = (df[df['target_rank_clean'] == 1.0].groupby('jockey_trainer_combo')['target_rank_clean'].count() / df.groupby('jockey_trainer_combo')['target_rank_clean'].count()).to_dict()
jockey_stats = (df[df['target_rank_clean'] == 1.0].groupby('騎手')['target_rank_clean'].count() / df.groupby('騎手')['target_rank_clean'].count()).to_dict()
df['trainer_win_rate'] = df['trainer_clean'].map(trainer_stats).fillna(0.05)
df['combo_win_rate'] = df['jockey_trainer_combo'].map(combo_stats).fillna(0.05)
df['jockey_win_rate'] = df['騎手'].map(jockey_stats).fillna(0.05)

# 特徴量リスト（格付け・クラス比較を追加した29指標）
features = [
    'horse_prize_avg', 'race_prize_relative', 'race_prize_rank',
    'is_minami_kanto', 'prev_is_minami', 'recent_avg_rank_3', 'recent_avg_rank_5', 
    'same_dist_avg_rank', 'same_place_avg_rank', 'days_since_prev', 'is_large_weight_change',
    'prev_1c', 'last_corner', 'corner_diff', 'last_3f_avg_rank', 'avg_time_diff', 'bad_baba_avg_rank', 'is_bad_baba',
    'horse_career_runs', 'jockey_win_rate', 'trainer_win_rate', 'combo_win_rate',
    '斤量', 'body_weight', 'kinryo_weight_ratio', 'distance_num',
    'race_front_runners', 'waku_win_rate'
]

X = df[features].fillna(0.0).astype(float)
y_relevance = df['relevance']

# レースごとの頭数（グループ単位）
groups = df.groupby('race_id', sort=False).size().values

print(f"✨ 全 {len(df)} 件 / {len(groups)} レースのグループ構造で Ranking モデルを学習します...")

# 🌟 LambdaMART Ranking モデルの学習
print("\n--- 【1/3】 LightGBM Ranker (LambdaMART) 学習中... ---")
ranker_lgb = lgb.LGBMRanker(
    objective='lambdarank',
    metric='ndcg',
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42
)
ranker_lgb.fit(X, y_relevance, group=groups)

print("\n--- 【2/3】 XGBoost Ranker (rank:ndcg) 学習中... ---")
ranker_xgb = xgb.XGBRanker(
    objective='rank:ndcg',
    n_estimators=200,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)
ranker_xgb.fit(X, y_relevance, group=groups)

print("\n--- 【3/3】 CatBoost Ranker (YetiRank) 学習中... ---")
ranker_cat = cb.CatBoostRanker(
    loss_function='YetiRank',
    iterations=200,
    learning_rate=0.03,
    depth=5,
    random_state=42,
    verbose=False
)
ranker_cat.fit(X, y_relevance, group_id=df['race_id'])

joblib.dump({
    'model_rank_lgb': ranker_lgb,
    'model_rank_xgb': ranker_xgb,
    'model_rank_cat': ranker_cat,
    'features': features
}, MODEL_FILE)

print(f"\n✨ 反映完了: Ranking(LambdaMART)モデル（{MODEL_FILE}）の出力が完了しました！")