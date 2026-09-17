import re
import joblib
import pandas as pd
import numpy as np
import unicodedata

# ① 蓋をしない・スキップしない
# ファイルがなければPythonのFileNotFoundErrorで正しくクラッシュさせる
df_future = pd.read_csv("future_races_chiho.csv", encoding='utf-8-sig')
df_past = pd.read_csv("ml_target_data_chiho.csv", encoding='utf-8-sig')
model_data = joblib.load("keiba_ai_model_nar_ensemble.pkl")
loaded_dicts = joblib.load("past_dicts.pkl")

features = model_data['features']
jockey_dict = loaded_dicts['jockey_dict']
jockey_rentai_dict = loaded_dicts['jockey_rentai_dict']

def clean_horse_name(name_val): 
    # NaNや空白の判定をごまかさない。文字列として処理し、クリーニングする。
    s = unicodedata.normalize('NFKC', str(name_val))
    s = re.sub(r'\(.*?\)|\[.*?\]|（.*?）|［.*?］', '', s)
    s = re.sub(r'[\s・･._\u3000\t\r\n]+', '', s)
    return s.strip().upper()

def get_kyakushitsu(fc): 
    # try-exceptで蓋をしない。パースエラーが起きるならデータがおかしい証拠として止める。
    fc_val = float(fc)
    if fc_val <= 2.5: return "逃"
    elif fc_val <= 4.5: return "先"
    elif fc_val <= 7.5: return "差"
    else: return "追"

def calc_track_bias(place_code, track_cond, kyaku):
    bonus = 0.0
    is_wet = track_cond in ['重', '不良']
    if is_wet and kyaku in ['逃', '先']: bonus += 1.0
    if track_cond == '稍重' and kyaku in ['逃', '先']: bonus += 0.5

    if place_code == '44': 
        if is_wet and kyaku == '逃': bonus += 1.5
        elif track_cond == '良' and kyaku == '差': bonus += 0.5
    elif place_code == '54':
        if is_wet:
            if kyaku in ['差', '追']: bonus += 2.0
            elif kyaku == '逃': bonus -= 1.0
    elif place_code in ['42', '45']:
        if is_wet and kyaku == '逃': bonus += 2.0
        elif is_wet and kyaku == '先': bonus += 1.0
    elif place_code == '30':
        if is_wet and kyaku in ['逃', '先']: bonus += 1.5
    return round(bonus, 1)

def main():
    # キーの前処理
    horse_col_f = '馬名' if '馬名' in df_future.columns else df_future.columns[0]
    df_future['馬名_clean'] = df_future[horse_col_f].astype(str).apply(clean_horse_name)
    df_future['race_id_clean'] = df_future['race_id'].astype(str)
    df_future['place_code_str'] = df_future['race_id_clean'].str[4:6]

    jockey_col_f = '騎手' if '騎手' in df_future.columns else 'jockey'
    df_future['騎手_clean'] = df_future[jockey_col_f].astype(str).apply(clean_horse_name)

    df_calc = df_future.copy()
    
    # 辞書から直接マッピング (デフォルト値0.05等で蓋をしない。見つからなければNaNになる)
    df_calc['jockey_win_rate'] = df_calc['騎手_clean'].map(jockey_dict)
    df_calc['jockey_rentai_rate'] = df_calc['騎手_clean'].map(jockey_rentai_dict)

    horse_col_p = '馬名_clean' if '馬名_clean' in df_past.columns else '馬名'
    df_past['馬名_clean'] = df_past[horse_col_p].astype(str).apply(clean_horse_name)
    df_past_latest = df_past.groupby('馬名_clean').last().reset_index()
    
    cols_to_drop = [c for c in df_past_latest.columns if c in df_calc.columns and c != '馬名_clean']
    df_past_clean = df_past_latest.drop(columns=cols_to_drop)
    
    # マージ
    df_calc = pd.merge(df_calc, df_past_clean, on='馬名_clean', how='left')

    df_calc['rank_score_raw'] = np.nan
    df_calc['score_disp_base'] = np.nan
    df_calc['脚質'] = pd.Series(None, index=df_calc.index, dtype=object)

    relative_bases = {f[:-10] for f in features if f.endswith('_race_diff')} | \
                     {f[:-12] for f in features if f.endswith('_race_zscore')} | \
                     {f[:-10] for f in features if f.endswith('_race_rank')}

    # 推論処理
    for rid in df_calc['race_id_clean'].unique():
        mask = df_calc['race_id_clean'] == rid
        race_df = df_calc[mask].copy()
        
        for base_col in relative_bases:
            vals = pd.to_numeric(race_df[base_col])
            mean_v, std_v = vals.mean(), vals.std(ddof=0)
            if pd.isna(std_v) or std_v == 0: std_v = 1.0
            
            diff = vals - mean_v
            df_calc.loc[mask, f'{base_col}_race_diff'] = diff
            df_calc.loc[mask, f'{base_col}_race_zscore'] = diff / std_v
            df_calc.loc[mask, f'{base_col}_race_rank'] = vals.rank(ascending=False, method='min')

        X_pred = df_calc.loc[mask, features].astype(float)

        preds = []
        for m in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat']:
            if m in model_data and model_data[m] is not None:
                preds.append(model_data[m].predict(X_pred))

        if preds:
            raw_scores = np.mean(preds, axis=0)
            score_mean = np.nanmean(raw_scores)
            score_std = np.nanstd(raw_scores, ddof=0)
            if pd.isna(score_std) or score_std == 0: score_std = 1.0
            
            base_score = np.round(((raw_scores - score_mean) / score_std) * 10 + 50, 1)
            df_calc.loc[mask, 'rank_score_raw'] = raw_scores
            df_calc.loc[mask, 'score_disp_base'] = base_score

        # 脚質判定
        if 'prev1_1c' in race_df.columns:
            df_calc.loc[mask, '脚質'] = race_df['prev1_1c'].apply(get_kyakushitsu).values

        # 馬場ボーナス事前計算
        for cond in ['良', '稍重', '重', '不良']:
            bonus_col = f'track_bonus_{cond}'
            score_col = f'score_{cond}'
            bonuses = []
            
            for i, row_data in df_calc[mask].iterrows():
                kyaku = row_data['脚質']
                place = row_data['place_code_str']
                b = calc_track_bias(place, cond, kyaku)
                bonuses.append(b)
                
            df_calc.loc[mask, bonus_col] = bonuses
            df_calc.loc[mask, score_col] = df_calc.loc[mask, 'score_disp_base'] + bonuses

    # データをそのまま出力（fillnaによる隠蔽を全削除）
    cache_data = {}
    for rid in df_calc['race_id_clean'].unique():
        race_final = df_calc[df_calc['race_id_clean'] == rid].copy()
        race_final['sort_key'] = pd.to_numeric(race_final['rank_score_raw'])
        race_final = race_final.sort_values(by=['sort_key'], ascending=False)
        cache_data[str(rid)] = race_final.to_dict('records')

    joblib.dump(cache_data, "app_cache_chiho.pkl")

if __name__ == "__main__":
    main()