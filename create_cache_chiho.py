import re
import sys
import joblib
import pandas as pd
import numpy as np
import unicodedata

# 1. データ・モデルのロード
df_future = pd.read_csv("future_races_chiho.csv", encoding='utf-8-sig')
df_past = pd.read_csv("ml_target_data_chiho.csv", encoding='utf-8-sig')
model_data = joblib.load("keiba_ai_model_nar_ensemble.pkl")
loaded_dicts = joblib.load("past_dicts.pkl")

features = model_data['features']
jockey_dict = loaded_dicts['jockey_dict']
jockey_rentai_dict = loaded_dicts['jockey_rentai_dict']

def clean_horse_name(name_val): 
    if pd.isna(name_val) or str(name_val).strip().lower() in ['nan', 'none', '']: return ""
    s = unicodedata.normalize('NFKC', str(name_val))
    s = re.sub(r'\(.*?\)|\[.*?\]|（.*?）|［.*?］', '', s)
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def get_kyakushitsu(fc): 
    try:
        fc_val = float(fc)
        if fc_val <= 2.5: return "逃"
        elif fc_val <= 4.5: return "先"
        elif fc_val <= 7.5: return "差"
        else: return "追"
    except:
        return "-"

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
    # ----------------------------------------------------
    # STEP 1: 先頭でのデータクレンジングと完全事前結合
    # ----------------------------------------------------
    horse_col_f = '馬名' if '馬名' in df_future.columns else df_future.columns[0]
    df_future['馬名_clean'] = df_future[horse_col_f].astype(str).apply(clean_horse_name)
    df_future['race_id_clean'] = df_future['race_id'].astype(str)
    df_future['place_code_str'] = df_future['race_id_clean'].str[4:6]

    jockey_col_f = '騎手' if '騎手' in df_future.columns else 'jockey'
    df_future['騎手_clean'] = df_future[jockey_col_f].astype(str).apply(clean_horse_name)

    # 過去データの馬名正規化と最新走抽出
    horse_col_p = '馬名_clean' if '馬名_clean' in df_past.columns else '馬名'
    df_past['馬名_clean'] = df_past[horse_col_p].astype(str).apply(clean_horse_name)
    df_past_latest = df_past.groupby('馬名_clean').last().reset_index()

    # 結合時の列名衝突（_x, _y 化）を防ぐため、キー以外の重複列を事前に全て排除
    overlap_cols = [c for c in df_past_latest.columns if c in df_future.columns and c != '馬名_clean']
    df_past_clean = df_past_latest.drop(columns=overlap_cols)
    df_past_clean['_merged_from_past'] = True

    # ★ ここで最優先で2つのCSVを結合 ★
    df_calc = pd.merge(df_future, df_past_clean, on='馬名_clean', how='left')
    df_calc['_merged_from_past'] = df_calc['_merged_from_past'].fillna(False)

    # 実データ不在馬（完全な新馬・初出走馬）の定義
    df_calc['is_no_data_horse'] = ~df_calc['_merged_from_past']

    # 騎手勝率等の事前マップ
    df_calc['jockey_win_rate'] = df_calc['騎手_clean'].map(jockey_dict).fillna(0.05)
    df_calc['jockey_rentai_rate'] = df_calc['騎手_clean'].map(jockey_rentai_dict).fillna(0.10)

    # モデルに必要なカラムが存在しない場合はNaN補填
    for col in features:
        if col not in df_calc.columns:
            df_calc[col] = np.nan

    df_calc['rank_score_raw'] = np.nan
    df_calc['score_disp_base'] = np.nan
    df_calc['脚質'] = pd.Series(None, index=df_calc.index, dtype=object)

    # ----------------------------------------------------
    # STEP 2: 結合済みデータに対する特徴量計算・モデル推論
    # ----------------------------------------------------
    relative_bases = {f[:-10] for f in features if f.endswith('_race_diff')} | \
                     {f[:-12] for f in features if f.endswith('_race_zscore')} | \
                     {f[:-10] for f in features if f.endswith('_race_rank')}

    for rid in df_calc['race_id_clean'].unique():
        mask = df_calc['race_id_clean'] == rid
        valid_data_mask = mask & (~df_calc['is_no_data_horse'])
        
        # 過去データが存在する馬のみでレース内標準化・相対化を実行
        for base_col in relative_bases:
            if base_col in df_calc.columns:
                vals = pd.to_numeric(df_calc.loc[valid_data_mask, base_col], errors='coerce')
                mean_v = vals.mean(skipna=True)
                std_v = vals.std(ddof=0, skipna=True)
                if pd.isna(std_v) or std_v == 0: std_v = 1.0
                
                diff = pd.to_numeric(df_calc.loc[mask, base_col], errors='coerce') - (mean_v if pd.notna(mean_v) else 0)
                df_calc.loc[mask, f'{base_col}_race_diff'] = diff
                df_calc.loc[mask, f'{base_col}_race_zscore'] = diff / std_v
                df_calc.loc[mask, f'{base_col}_race_rank'] = pd.to_numeric(df_calc.loc[mask, base_col], errors='coerce').rank(ascending=False, method='min', na_option='bottom')

        # 実データが存在する馬のみ推論
        valid_indices = df_calc[valid_data_mask].index
        if len(valid_indices) > 0:
            X_pred = df_calc.loc[valid_indices, features].copy()
            for col in features:
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')

            preds = []
            for m in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat']:
                if m in model_data and model_data[m] is not None:
                    try:
                        p = model_data[m].predict(X_pred)
                        preds.append(p)
                    except Exception:
                        pass

            if preds:
                raw_scores = np.nanmean(preds, axis=0)
                score_mean = np.nanmean(raw_scores)
                score_std = np.nanstd(raw_scores, ddof=0)
                if pd.isna(score_std) or score_std == 0: score_std = 1.0
                
                z_scores = np.round((raw_scores - score_mean) / score_std, 2)
                df_calc.loc[valid_indices, 'rank_score_raw'] = raw_scores
                df_calc.loc[valid_indices, 'score_disp_base'] = z_scores

        race_df = df_calc[mask].copy()
        if 'prev1_1c' in race_df.columns:
            df_calc.loc[mask, '脚質'] = race_df['prev1_1c'].apply(get_kyakushitsu).values

        # 馬場ボーナス計算
        for cond in ['良', '稍重', '重', '不良']:
            bonus_col = f'track_bonus_{cond}'
            score_col = f'score_{cond}'
            bonuses = []
            
            for i, row_data in df_calc[mask].iterrows():
                kyaku = row_data['脚質']
                place = str(row_data['place_code_str'])
                b = calc_track_bias(place, cond, kyaku)
                bonuses.append(b)
                
            df_calc.loc[mask, bonus_col] = bonuses
            
            # データ無の馬はスコア計算を行わず NaN に維持（表示側で「データ無」）
            base_s = pd.to_numeric(df_calc.loc[mask, 'score_disp_base'], errors='coerce')
            df_calc.loc[mask, score_col] = base_s + bonuses

    # ----------------------------------------------------
    # STEP 3: キャッシュ保存
    # ----------------------------------------------------
    cache_data = {}
    for rid in df_calc['race_id_clean'].unique():
        race_final = df_calc[df_calc['race_id_clean'] == rid].copy()
        race_final['sort_key'] = pd.to_numeric(race_final['rank_score_raw']).fillna(-999)
        race_final = race_final.sort_values(by=['sort_key'], ascending=False)
        cache_data[str(rid)] = race_final.to_dict('records')

    joblib.dump(cache_data, "app_cache_chiho.pkl")

if __name__ == "__main__":
    main()