import re
import sys
import joblib
import pandas as pd
import numpy as np
import unicodedata

def log_header(title):
    print(f"\n==================================================")
    print(f" ⚙️  {title}")
    print(f"==================================================")

def clean_horse_name(name_val): 
    if pd.isna(name_val) or str(name_val).strip().lower() in ['nan', 'none', '']: return ""
    s = unicodedata.normalize('NFKC', str(name_val))
    s = re.sub(r'\(.*?\)|\[.*?\]|（.*?）|［.*?］', '', s)
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def get_kyakushitsu(fc): 
    if pd.isna(fc) or str(fc).strip().lower() in ['nan', 'none', '', '-']:
        return "-"
    try:
        fc_val = float(fc)
        if fc_val <= 2.5: return "逃"
        elif fc_val <= 4.5: return "先"
        elif fc_val <= 7.5: return "差"
        else: return "追"
    except:
        return "-"

def calc_track_bias(place_code, track_cond, kyaku):
    if kyaku == "-": return 0.0
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
    log_header("1. データセットおよび学習モデルの読み込み")
    try:
        df_future = pd.read_csv("future_races_chiho.csv", encoding='utf-8-sig')
        df_past = pd.read_csv("ml_target_data_chiho.csv", encoding='utf-8-sig')
        model_data = joblib.load("keiba_ai_model_nar_ensemble.pkl")
        loaded_dicts = joblib.load("past_dicts.pkl")
        print(f"  [SUCCESS] 出走予定データ : {len(df_future)} 件 (全 {df_future['race_id'].nunique()} レース)")
        print(f"  [SUCCESS] 過去実績DBデータ : {len(df_past)} 件")
    except Exception as e:
        print(f"  [CRITICAL ERROR] ファイル読み込み失敗: {e}")
        sys.exit(1)

    features = model_data['features']
    jockey_dict = loaded_dicts['jockey_dict']
    jockey_rentai_dict = loaded_dicts['jockey_rentai_dict']

    log_header("2. 馬名クレンジング & 過去データ最優先結合")
    horse_col_f = '馬名' if '馬名' in df_future.columns else df_future.columns[0]
    df_future['馬名_clean'] = df_future[horse_col_f].astype(str).apply(clean_horse_name)
    df_future['race_id_clean'] = df_future['race_id'].astype(str)
    df_future['place_code_str'] = df_future['race_id_clean'].str[4:6]

    jockey_col_f = '騎手' if '騎手' in df_future.columns else 'jockey'
    df_future['騎手_clean'] = df_future[jockey_col_f].astype(str).apply(clean_horse_name)

    horse_col_p = '馬名_clean' if '馬名_clean' in df_past.columns else '馬名'
    df_past['馬名_clean'] = df_past[horse_col_p].astype(str).apply(clean_horse_name)
    df_past_latest = df_past.groupby('馬名_clean').last().reset_index()

    overlap_cols = [c for c in df_past_latest.columns if c in df_future.columns and c != '馬名_clean']
    df_past_clean = df_past_latest.drop(columns=overlap_cols)
    df_past_clean['_merged_from_past'] = True

    df_calc = pd.merge(df_future, df_past_clean, on='馬名_clean', how='left')
    
    # bool型判定を厳格化（符号反転バグの完全対策）
    df_calc['_merged_from_past'] = df_calc['_merged_from_past'].fillna(False).astype(bool)
    df_calc['is_no_data_horse'] = ~df_calc['_merged_from_past']

    matched_count = int(df_calc['_merged_from_past'].sum())
    no_data_count = int(df_calc['is_no_data_horse'].sum())
    total_horses = len(df_calc)
    match_rate = (matched_count / total_horses) * 100 if total_horses > 0 else 0

    print(f"  * 総出走頭数     : {total_horses} 頭")
    print(f"  * 過去データ有   : {matched_count} 頭 ({match_rate:.1f}%)")
    print(f"  * 過去データ無   : {no_data_count} 頭")

    # 未紐付け馬名を安全に取得して画面表示
    target_name_col = horse_col_f if horse_col_f in df_calc.columns else '馬名_clean'
    unmatched_names = df_calc.loc[df_calc['is_no_data_horse'], target_name_col].unique().tolist()
    print(f"\n  🔍 【未紐付け馬一覧】 (全 {len(unmatched_names)} 頭)")
    print(f"  {unmatched_names}\n")

    df_calc['jockey_win_rate'] = df_calc['騎手_clean'].map(jockey_dict).fillna(0.05)
    df_calc['jockey_rentai_rate'] = df_calc['騎手_clean'].map(jockey_rentai_dict).fillna(0.10)

    for col in features:
        if col not in df_calc.columns:
            df_calc[col] = np.nan

    df_calc['rank_score_raw'] = np.nan
    df_calc['score_disp_base'] = np.nan
    df_calc['脚質'] = pd.Series(None, index=df_calc.index, dtype=object)

    log_header("3. レース別AI推論 & 特徴量標準化 (Zスコア)")
    relative_bases = {f[:-10] for f in features if f.endswith('_race_diff')} | \
                     {f[:-12] for f in features if f.endswith('_race_zscore')} | \
                     {f[:-10] for f in features if f.endswith('_race_rank')}

    race_stats = {"normal": 0, "minor_missing": 0, "skip_recommended": 0}

    for rid in df_calc['race_id_clean'].unique():
        mask = df_calc['race_id_clean'] == rid
        race_no_data_num = int(df_calc.loc[mask, 'is_no_data_horse'].sum())

        if race_no_data_num == 0: race_stats["normal"] += 1
        elif race_no_data_num < 3: race_stats["minor_missing"] += 1
        else: race_stats["skip_recommended"] += 1

        valid_data_mask = mask & (~df_calc['is_no_data_horse'])
        
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
        else:
            df_calc.loc[mask, '脚質'] = "-"

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
            base_s = pd.to_numeric(df_calc.loc[mask, 'score_disp_base'], errors='coerce')
            df_calc.loc[mask, score_col] = base_s + bonuses

    print(f"  * 正常レース (データ無 0頭)      : {race_stats['normal']} レース")
    print(f"  * 軽微欠損 (データ無 1〜2頭)     : {race_stats['minor_missing']} レース")
    print(f"  * ⚠️ 見送り推奨 (データ無 3頭以上) : {race_stats['skip_recommended']} レース")

    log_header("4. アプリ用キャッシュファイルの生成")
    cache_data = {}
    for rid in df_calc['race_id_clean'].unique():
        race_final = df_calc[df_calc['race_id_clean'] == rid].copy()
        race_final['sort_key'] = pd.to_numeric(race_final['rank_score_raw']).fillna(-999)
        race_final = race_final.sort_values(by=['sort_key'], ascending=False)
        cache_data[str(rid)] = race_final.to_dict('records')

    joblib.dump(cache_data, "app_cache_chiho.pkl")
    print(f"  [SUCCESS] 'app_cache_chiho.pkl' を正常に出力しました。 (全 {len(cache_data)} レース格納)")
    print("==================================================\n")

if __name__ == "__main__":
    main()