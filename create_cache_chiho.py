import os
import re
import joblib
import pandas as pd
import numpy as np
import unicodedata

CSV_FUTURE = "future_races_chiho.csv"
CSV_PAST_V2 = "ml_target_data_chiho_v2.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"
DICT_FILE = "past_dicts.pkl"
CACHE_FILE = "app_cache_chiho.pkl"

def clean_horse_name(name_val): 
    if pd.isna(name_val): return ""
    return re.sub(r'[\s・･._\u3000\t\r\n]+', '', unicodedata.normalize('NFKC', str(name_val))).strip()

def get_kyakushitsu(fc): 
    return "逃" if fc <= 2.5 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

model_data = joblib.load(MODEL_FILE)
features = model_data.get('features', [])

# 1. 辞書のロード (騎手・調教師データのみ使用)
loaded_dicts = joblib.load(DICT_FILE) if os.path.exists(DICT_FILE) else {}
jockey_dict = loaded_dicts.get('jockey_dict', {})
trainer_dict = loaded_dicts.get('trainer_dict', {})
jockey_rentai_dict = loaded_dicts.get('jockey_rentai_dict', {})

# 2. 過去データV2のロード
df_past = pd.read_csv(CSV_PAST_V2, low_memory=False)
horse_col = '馬名' if '馬名' in df_past.columns else 'horse_name'
df_past['馬名_clean'] = df_past[horse_col].astype(str).apply(clean_horse_name)

date_col = 'date_parsed' if 'date_parsed' in df_past.columns else 'date'
df_past['date_dt'] = pd.to_datetime(df_past[date_col], errors='coerce')

for col in features:
    if col in df_past.columns:
        if df_past[col].dtype == object or pd.api.types.is_string_dtype(df_past[col]):
            df_past[col] = df_past[col].astype(str).str.replace(',', '', regex=False)
        df_past[col] = pd.to_numeric(df_past[col], errors='coerce')

df_past = df_past.sort_values(['馬名_clean', 'date_dt'])
num_cols = df_past.select_dtypes(include=[np.number]).columns.tolist()
df_past[num_cols] = df_past.groupby('馬名_clean')[num_cols].ffill()
df_past_latest = df_past.groupby('馬名_clean').last().reset_index()

# 3. 出馬表ロード
df_future = pd.read_csv(CSV_FUTURE, low_memory=False)
df_future['馬名_clean'] = df_future['馬名'].astype(str).apply(clean_horse_name)
df_future['騎手_clean'] = df_future['騎手'].astype(str).apply(clean_horse_name)
df_future['trainer_clean'] = df_future.get('調教師', df_future['騎手']).astype(str).apply(clean_horse_name)

df_future['kinryo_num'] = pd.to_numeric(df_future.get('斤量'), errors='coerce').fillna(54.0)
df_future['斤量'] = df_future['kinryo_num']
df_future['race_id_clean'] = pd.to_numeric(df_future['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)

if '距離' in df_future.columns and 'distance' not in df_future.columns:
    df_future['distance'] = df_future['距離']
if 'distance' in df_future.columns:
    df_future['distance_num'] = pd.to_numeric(df_future['distance'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    df_future['distance'] = df_future['distance_num']

if '馬番' in df_future.columns:
    df_future['gate_num'] = pd.to_numeric(df_future['馬番'], errors='coerce')
if '枠番' in df_future.columns:
    df_future['waku_num'] = pd.to_numeric(df_future['枠番'], errors='coerce')

if '馬体重' in df_future.columns:
    def parse_weight_info(val):
        m = re.search(r'(\d{3})(?:\(([-+]?\d+)\))?', str(val).strip())
        if m: return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0
        return 470.0, 0.0
    parsed_w = df_future['馬体重'].apply(parse_weight_info)
    df_future['body_weight'] = parsed_w.apply(lambda x: x[0])
    df_future['weight_change'] = parsed_w.apply(lambda x: x[1])
else:
    df_future['body_weight'], df_future['weight_change'] = 470.0, 0.0

df_future['weight_num'] = df_future['body_weight']
df_future['horse_weight'] = df_future['body_weight']
df_future['kinryo_weight_ratio'] = df_future['kinryo_num'] / df_future['body_weight']

# 💡 騎手・調教師の勝率だけは辞書から取得し、問題の dict_feature_mapping は削除！
df_future['jockey_win_rate'] = df_future['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
df_future['jockey_rentai_rate'] = df_future['騎手_clean'].apply(lambda x: jockey_rentai_dict.get(x, 0.10))
df_future['trainer_win_rate'] = df_future['trainer_clean'].apply(lambda x: trainer_dict.get(x, 0.05))

# 4. 出馬表＋補完用の過去データをマージ
cols_to_drop_from_past = [c for c in df_past_latest.columns if c in df_future.columns and c != '馬名_clean']
df_past_clean = df_past_latest.drop(columns=cols_to_drop_from_past)
df_merged = pd.merge(df_future, df_past_clean, on='馬名_clean', how='left')

time_idx_cols = ['eff_my_time_idx', 'prev_my_time_idx', 'best_time_idx']
start_idx_cols = ['eff_my_start_idx', 'prev_my_start_idx']

for c in time_idx_cols:
    if c in df_merged.columns: df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce').fillna(100.0)
for c in start_idx_cols:
    if c in df_merged.columns: df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce').fillna(50.0)

# 5. 推論 & キャッシュ生成
cache_data = {}
races = df_merged['race_id_clean'].unique()
base_diff_cols = [c.replace('_race_diff', '') for c in features if c.endswith('_race_diff')]

for rid in races:
    if str(rid) == '0': continue
    race_df = df_merged[df_merged['race_id_clean'] == rid].copy()
    if race_df.empty: continue

    for col in features:
        if col in race_df.columns:
            if race_df[col].dtype == object or pd.api.types.is_string_dtype(race_df[col]):
                race_df[col] = race_df[col].astype(str).str.replace(',', '', regex=False)
            race_df[col] = pd.to_numeric(race_df[col], errors='coerce')

    for col in base_diff_cols:
        if col in race_df.columns:
            mean_v = race_df[col].mean(skipna=True)
            std_v = race_df[col].std(ddof=0, skipna=True)
            if pd.isna(mean_v): mean_v = 0.0
            if pd.isna(std_v) or std_v == 0: std_v = 1.0  
            race_df[f'{col}_race_diff'] = race_df[col] - mean_v
            race_df[f'{col}_race_zscore'] = (race_df[col] - mean_v) / std_v

    X_pred = pd.DataFrame(index=race_df.index)
    for col in features:
        if col in race_df.columns:
            X_pred[col] = race_df[col].fillna(0.0)
        else:
            X_pred[col] = 0.0

    X_pred = X_pred.astype(float)

    preds = []
    for m_key in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat']:
        if m_key in model_data and model_data[m_key] is not None:
            preds.append(model_data[m_key].predict(X_pred))
            
    if not preds and 'model' in model_data:
        preds.append(model_data['model'].predict(X_pred))

    if preds:
        race_df['rank_score_raw'] = np.mean(preds, axis=0)
        score_mean = race_df['rank_score_raw'].mean(skipna=True)
        score_std = race_df['rank_score_raw'].std(ddof=0, skipna=True)
        if pd.isna(score_std) or score_std == 0: score_std = 1.0
        race_df['score_disp'] = np.round(((race_df['rank_score_raw'] - score_mean) / score_std) * 10 + 50).astype(int)
    else:
        race_df['score_disp'] = 50
        race_df['rank_score_raw'] = 0.0

    col_kyakushitsu = 'prev_1c' if 'prev_1c' in race_df.columns else ('first_corner' if 'first_corner' in race_df.columns else None)
    if col_kyakushitsu:
        race_df['脚質'] = pd.to_numeric(race_df[col_kyakushitsu], errors='coerce').fillna(5.0).apply(get_kyakushitsu)
    else:
        race_df['脚質'] = "-"

    if 'date_dt' in race_df.columns:
        race_df['date_dt'] = race_df['date_dt'].astype(str).replace('NaT', '')
        
    for col in race_df.columns:
        if pd.api.types.is_numeric_dtype(race_df[col]):
            race_df[col] = race_df[col].fillna(0.0)
        else:
            race_df[col] = race_df[col].fillna("")

    race_df = race_df.sort_values(by=['rank_score_raw'], ascending=False).reset_index(drop=True)
    cache_data[str(rid)] = race_df.to_dict('records')

joblib.dump(cache_data, CACHE_FILE)
print("✨ AIが本来の性能を発揮できる、完璧なキャッシュデータを出力しました！")