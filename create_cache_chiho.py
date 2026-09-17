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

def get_kyakushitsu(row): 
    fc = row.get('prev1_1c', np.nan)
    tosu = row.get('prev1_tosu', np.nan)
    
    if pd.isna(fc) or str(fc).strip().lower() in ['nan', 'none', '', '-']:
        return "-"
    try:
        fc_val = float(fc)
        if fc_val <= 1.0: 
            return "逃"
        elif fc_val <= 3.0: 
            return "先"
        elif pd.notna(tosu) and float(tosu) > 0:
            ratio = fc_val / float(tosu)
            if ratio <= 0.6: return "差"
            else: return "追"
        else:
            if fc_val <= 6.0: return "差"
            else: return "追"
    except:
        return "-"

def calc_track_bias(place_code, track_cond, kyaku):
    if kyaku == "-": return 0.0
    bonus = 0.0
    is_wet = track_cond in ['重', '不良']
    if is_wet and kyaku == '逃': bonus += 2.0
    elif is_wet and kyaku == '先': bonus += 1.0
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

    log_header("2. 安全な列選別マージ (当日の出走表データマスター化)")
    
    # 馬名・騎手・レースIDの整形
    horse_col_f = '馬名' if '馬名' in df_future.columns else df_future.columns[0]
    df_future['馬名_clean'] = df_future[horse_col_f].astype(str).apply(clean_horse_name)
    df_future['race_id_clean'] = df_future['race_id'].astype(str)
    df_future['place_code_str'] = df_future['race_id_clean'].str[4:6]

    jockey_col_f = '騎手' if '騎手' in df_future.columns else 'jockey'
    df_future['騎手_clean'] = df_future[jockey_col_f].astype(str).apply(clean_horse_name)

    # 過去データから「馬の過去実績指標」のみを厳選抽出（当日情報との衝突を完全遮断）
    horse_col_p = '馬名_clean' if '馬名_clean' in df_past.columns else '馬名'
    df_past['馬名_clean'] = df_past[horse_col_p].astype(str).apply(clean_horse_name)
    df_past_latest = df_past.groupby('馬名_clean').last().reset_index()

    # 当日データと重複するカラム、および過去のレース内相対計算カラム（_race_zscore等）を除外
    exclude_keywords = ['_race_diff', '_race_zscore', '_race_rank', 'race_id', '枠番', '馬番', '斤量', '騎手', 'オッズ', '人気', '着順', 'target_win', 'target_place']
    past_keep_cols = ['馬名_clean']
    for c in df_past_latest.columns:
        if c not in df_future.columns and not any(k in c for k in exclude_keywords):
            past_keep_cols.append(c)

    df_past_clean = df_past_latest[past_keep_cols].copy()
    df_past_clean['_merged_from_past'] = True

    # 主軸(df_future)に過去実績プロファイルのみを left join
    df_calc = pd.merge(df_future, df_past_clean, on='馬名_clean', how='left')
    df_calc['_merged_from_past'] = df_calc['_merged_from_past'].fillna(False).astype(bool)

    # 過去DB不在、またはタイム指数未保有の馬を「データ無馬」としてフラグ化
    if 'ema3_custom_time_index_m' in df_calc.columns:
        has_valid_idx = df_calc['ema3_custom_time_index_m'].notna() & (pd.to_numeric(df_calc['ema3_custom_time_index_m'], errors='coerce') > 0)
        df_calc['is_no_data_horse'] = (~df_calc['_merged_from_past']) | (~has_valid_idx)
    else:
        df_calc['is_no_data_horse'] = ~df_calc['_merged_from_past']

    matched_count = int((~df_calc['is_no_data_horse']).sum())
    no_data_count = int(df_calc['is_no_data_horse'].sum())
    total_horses = len(df_calc)
    match_rate = (matched_count / total_horses) * 100 if total_horses > 0 else 0

    print(f"  * 総出走頭数     : {total_horses} 頭")
    print(f"  * 有効過去データ有 : {matched_count} 頭 ({match_rate:.1f}%)")
    print(f"  * 実質データ無し   : {no_data_count} 頭")

    target_name_col = horse_col_f if horse_col_f in df_calc.columns else '馬名_clean'
    unmatched_names = df_calc.loc[df_calc['is_no_data_horse'], target_name_col].unique().tolist()
    print(f"\n  🔍 【データ無（評価対象外）馬一覧】 (全 {len(unmatched_names)} 頭)")
    print(f"  {unmatched_names}\n")

    # 騎手勝率等の補填
    df_calc['jockey_win_rate'] = df_calc['騎手_clean'].map(jockey_dict).fillna(0.05)
    df_calc['jockey_rentai_rate'] = df_calc['騎手_clean'].map(jockey_rentai_dict).fillna(0.10)

    for col in features:
        if col not in df_calc.columns:
            df_calc[col] = np.nan

    df_calc['rank_score_raw'] = np.nan
    df_calc['score_disp_base'] = np.nan
    df_calc['脚質'] = pd.Series(None, index=df_calc.index, dtype=object)

    log_header("3. 当日レースメンバー内での相対評価再計算 & AI推論")
    
    # 当日のレース出走馬の中だけで相対値（_race_diff, _race_zscore）を再計算
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
        
        # 有効データ保持馬のみで平均・標準偏差を算出し、相対化
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

        # AI推論の実行
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
                max_score = np.nanmax(raw_scores)
                min_score = np.nanmin(raw_scores)
                
                # 60点〜100点スケールにスケーリング
                if max_score == min_score or pd.isna(max_score):
                    scores_100 = np.full_like(raw_scores, 80.0)
                else:
                    scores_100 = np.round(60 + (raw_scores - min_score) / (max_score - min_score) * 40, 1)

                df_calc.loc[valid_indices, 'rank_score_raw'] = raw_scores
                df_calc.loc[valid_indices, 'score_disp_base'] = scores_100

        # 脚質判定
        for idx in df_calc[mask].index:
            df_calc.loc[idx, '脚質'] = get_kyakushitsu(df_calc.loc[idx])

        # 馬場条件別の最終点数算出（最高100点）
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
            final_scores = np.minimum(100.0, base_s + bonuses)
            df_calc.loc[mask, score_col] = final_scores

    print(f"  * 正常レース (データ無 0頭)      : {race_stats['normal']} レース")
    print(f"  * 軽微欠損 (データ無 1〜2頭)     : {race_stats['minor_missing']} レース")
    print(f"  * ⚠️ 見送り推奨 (データ無 3頭以上) : {race_stats['skip_recommended']} レース")

    log_header("4. アプリ用キャッシュファイルの生成")
    cache_data = {}
    for rid in df_calc['race_id_clean'].unique():
        race_final = df_calc[df_calc['race_id_clean'] == rid].copy()
        # raw_score の降順で確実にソートして並べる
        race_final['sort_key'] = pd.to_numeric(race_final['rank_score_raw']).fillna(-999)
        race_final = race_final.sort_values(by=['sort_key'], ascending=False)
        cache_data[str(rid)] = race_final.to_dict('records')

    joblib.dump(cache_data, "app_cache_chiho.pkl")
    print(f"  [SUCCESS] 'app_cache_chiho.pkl' を正常に出力しました。 (全 {len(cache_data)} レース格納)")
    print("==================================================\n")

if __name__ == "__main__":
    main()