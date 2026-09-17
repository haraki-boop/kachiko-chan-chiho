import os
import re
import joblib
import pandas as pd
import numpy as np
import unicodedata

CSV_FUTURE = "future_races_chiho.csv"
CSV_PAST = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"
DICT_FILE = "past_dicts.pkl"
CACHE_FILE = "app_cache_chiho.pkl"

def clean_horse_name(name_val): 
    if pd.isna(name_val) or str(name_val).strip().lower() in ['nan', 'none', '']: return ""
    s = unicodedata.normalize('NFKC', str(name_val))
    s = re.sub(r'\(.*?\)|\[.*?\]|（.*?）|［.*?］', '', s)
    s = re.sub(r'[\s・･._\u3000\t\r\n]+', '', s)
    return s.strip().upper()

def get_kyakushitsu(fc): 
    try:
        fc_val = float(fc)
        return "逃" if fc_val <= 2.5 else "先" if fc_val <= 4.5 else "差" if fc_val <= 7.5 else "追"
    except: return "-"

def safe_read_csv(filepath):
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding=enc)
            if not df.empty: return df
        except: continue
    return pd.DataFrame()

def main():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(CSV_FUTURE): return

    model_data = joblib.load(MODEL_FILE)
    features = model_data.get('features', [])
    
    # 🚨復活: 辞書データの読み込み（連対率0%の解消）
    loaded_dicts = joblib.load(DICT_FILE) if os.path.exists(DICT_FILE) else {}
    jockey_dict = loaded_dicts.get('jockey_dict', {})
    trainer_dict = loaded_dicts.get('trainer_dict', {})
    jockey_rentai_dict = loaded_dicts.get('jockey_rentai_dict', {})

    df_future_master = safe_read_csv(CSV_FUTURE)
    if df_future_master.empty: return

    horse_col_f = '馬名' if '馬名' in df_future_master.columns else ('horse_name' if 'horse_name' in df_future_master.columns else df_future_master.columns[0])
    df_future_master['馬名_clean'] = df_future_master[horse_col_f].astype(str).apply(clean_horse_name)
    df_future_master['race_id_clean'] = pd.to_numeric(df_future_master['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)

    jockey_col_f = '騎手' if '騎手' in df_future_master.columns else 'jockey'
    if jockey_col_f in df_future_master.columns:
        df_future_master['騎手_clean'] = df_future_master[jockey_col_f].astype(str).apply(clean_horse_name)
    else:
        df_future_master['騎手_clean'] = ""

    df_calc = df_future_master.copy()

    # 🚨復活: 勝率・連対率のセット
    df_calc['jockey_win_rate'] = df_calc['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
    df_calc['jockey_rentai_rate'] = df_calc['騎手_clean'].apply(lambda x: jockey_rentai_dict.get(x, 0.10))

    df_past = safe_read_csv(CSV_PAST)
    if not df_past.empty:
        horse_col_p = '馬名_clean' if '馬名_clean' in df_past.columns else ('馬名' if '馬名' in df_past.columns else df_past.columns[0])
        df_past['馬名_clean'] = df_past[horse_col_p].astype(str).apply(clean_horse_name)
        df_past_latest = df_past.groupby('馬名_clean').last().reset_index()
        
        cols_to_drop = [c for c in df_past_latest.columns if c in df_calc.columns and c != '馬名_clean']
        df_past_clean = df_past_latest.drop(columns=cols_to_drop)
        
        df_calc = pd.merge(df_calc, df_past_clean, on='馬名_clean', how='left')

    relative_bases = {f[:-10] for f in features if f.endswith('_race_diff')} | \
                     {f[:-12] for f in features if f.endswith('_race_zscore')} | \
                     {f[:-10] for f in features if f.endswith('_race_rank')}

    ai_results = []
    races = df_calc['race_id_clean'].unique()

    for rid in races:
        if str(rid) == '0': continue
        race_df = df_calc[df_calc['race_id_clean'] == rid].copy()
        
        for base_col in relative_bases:
            if base_col in race_df.columns:
                vals = pd.to_numeric(race_df[base_col], errors='coerce')
                mean_v, std_v = vals.mean(skipna=True), vals.std(ddof=0, skipna=True)
                std_v = std_v if pd.notna(std_v) and std_v != 0 else 1.0
                race_df[f'{base_col}_race_diff'] = vals - (mean_v if pd.notna(mean_v) else 0)
                race_df[f'{base_col}_race_zscore'] = race_df[f'{base_col}_race_diff'] / std_v
                race_df[f'{base_col}_race_rank'] = vals.rank(ascending=False, method='min')

        X_pred = pd.DataFrame(index=race_df.index)
        for col in features:
            X_pred[col] = pd.to_numeric(race_df.get(col, np.nan), errors='coerce')

        preds = [model_data[m].predict(X_pred.astype(float)) for m in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat'] if m in model_data and model_data[m] is not None]

        if preds:
            race_df['rank_score_raw'] = np.mean(preds, axis=0)
            score_mean = race_df['rank_score_raw'].mean(skipna=True)
            score_std = race_df['rank_score_raw'].std(ddof=0, skipna=True)
            score_std = score_std if pd.notna(score_std) and score_std != 0 else 1.0
            race_df['score_disp'] = np.round(((race_df['rank_score_raw'] - (score_mean if pd.notna(score_mean) else 0)) / score_std) * 10 + 50, 1)
        else:
            race_df['rank_score_raw'] = np.nan
            race_df['score_disp'] = np.nan

        col_kyaku = 'prev1_1c' if 'prev1_1c' in race_df.columns else ('first_corner' if 'first_corner' in race_df.columns else None)
        race_df['脚質'] = race_df[col_kyaku].apply(get_kyakushitsu) if col_kyaku and col_kyaku in race_df.columns else "-"

        # 🚨修正: 過去の指数実績や適性データをすべて含めた状態で保存する
        ai_results.append(race_df)

    if ai_results:
        df_final = pd.concat(ai_results, ignore_index=True)
        # 🚨修正: 計算結果の「表示テキスト部分」だけを、絶対に崩れないマスターで安全に上書き復元する
        for col in ['馬番', '馬名', '性齢', '斤量', '騎手', '調教師', '馬体重', 'オッズ', '人気', '世論コメント']:
            if col in df_future_master.columns:
                df_final[col] = df_future_master[col]
    else:
        df_final = df_future_master.copy()
        df_final['rank_score_raw'] = np.nan
        df_final['score_disp'] = "-"
        df_final['脚質'] = "-"

    # フロント表示エラー防止
    df_final = df_final.fillna("-")

    cache_data = {}
    for rid in df_final['race_id_clean'].unique():
        if str(rid) == '0': continue
        race_final = df_final[df_final['race_id_clean'] == rid].copy()
        
        race_final['sort_key'] = pd.to_numeric(race_final['rank_score_raw'], errors='coerce')
        race_final = race_final.sort_values(by=['sort_key'], ascending=False, na_position='last').drop(columns=['sort_key'])
        
        cache_data[str(rid)] = race_final.to_dict('records')

    joblib.dump(cache_data, CACHE_FILE)
    print(f"✨ 完了: 全ての過去データ・実績指数を保持した状態で '{CACHE_FILE}' を作成しました。")

if __name__ == "__main__":
    main()