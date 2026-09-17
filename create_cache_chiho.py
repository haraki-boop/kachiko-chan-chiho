import os
import re
import joblib
import pandas as pd
import numpy as np
import unicodedata

CSV_FUTURE = "future_races_chiho.csv"
CSV_PAST = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"
CACHE_FILE = "app_cache_chiho.pkl"

def clean_horse_name(name_val): 
    if pd.isna(name_val) or str(name_val).strip().lower() in ['nan', 'none', '']: return ""
    s = unicodedata.normalize('NFKC', str(name_val))
    # 🚨 スクラッパーが拾いがちな (牡3) や [地] などのカッコ表記をすべて強制排除して純粋な馬名だけにする
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'（.*?）', '', s)
    s = re.sub(r'［.*?］', '', s)
    # 記号や空白を削除
    s = re.sub(r'[\s・･._\u3000\t\r\n]+', '', s)
    return s.strip().upper()

def get_kyakushitsu(fc): 
    try:
        fc_val = float(fc)
        return "逃" if fc_val <= 2.5 else "先" if fc_val <= 4.5 else "差" if fc_val <= 7.5 else "追"
    except:
        return "-"

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"🚨 {MODEL_FILE} が見つかりません。")
        return
    if not os.path.exists(CSV_FUTURE):
        print(f"🚨 {CSV_FUTURE} が見つかりません。")
        return

    model_data = joblib.load(MODEL_FILE)
    features = model_data.get('features', [])

    # 1. 出馬表ロード（最優先保持）
    df_future = pd.read_csv(CSV_FUTURE, low_memory=False)
    
    # 🚨 列名に全角スペース等が混入していても確実に「馬名」カラムを捕捉する
    horse_col_f = None
    for c in df_future.columns:
        c_clean = str(c).replace(' ', '').replace(' ', '').replace('\n', '')
        if c_clean in ['馬名', 'horse_name', 'horsename']:
            horse_col_f = c
            break
            
    if horse_col_f is None:
        print("🚨 [致命的エラー] 出馬表内に「馬名」カラムが見つかりません。")
        print(f"現在の列名: {df_future.columns.tolist()}")
        return

    df_future['馬名_display'] = df_future[horse_col_f].astype(str)
    df_future['馬名_clean'] = df_future['馬名_display'].apply(clean_horse_name)
    df_future['race_id_clean'] = pd.to_numeric(df_future['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)

    # 2. 過去データのロード＆マージ
    df_past = pd.read_csv(CSV_PAST, low_memory=False) if os.path.exists(CSV_PAST) else pd.DataFrame()
    
    if not df_past.empty:
        horse_col_p = '馬名_clean' if '馬名_clean' in df_past.columns else ('馬名' if '馬名' in df_past.columns else df_past.columns[0])
        df_past['馬名_clean'] = df_past[horse_col_p].astype(str).apply(clean_horse_name)
        df_past_latest = df_past.groupby('馬名_clean').last().reset_index()
        
        # 🚨 出馬表の元データ（馬名・騎手など）が上書き消去されるのを完全に防ぐ
        cols_to_drop = [c for c in df_past_latest.columns if c in df_future.columns and c != '馬名_clean']
        df_past_clean = df_past_latest.drop(columns=cols_to_drop)
        
        # ここで過去実績とガッチリ結合！
        df_merged = pd.merge(df_future, df_past_clean, on='馬名_clean', how='left')
    else:
        df_merged = df_future.copy()

    # 出馬表の表示用馬名を復旧
    df_merged['馬名'] = df_merged['馬名_display']

    # 3. レース単位での動的特徴量算出 (Diff, Zscore, Rank)
    cache_data = {}
    races = df_merged['race_id_clean'].unique()

    relative_bases = set()
    for f in features:
        if f.endswith('_race_diff'): relative_bases.add(f[:-10])
        elif f.endswith('_race_zscore'): relative_bases.add(f[:-12])
        elif f.endswith('_race_rank'): relative_bases.add(f[:-10])

    for rid in races:
        if str(rid) == '0': continue
        race_df = df_merged[df_merged['race_id_clean'] == rid].copy()
        if race_df.empty: continue

        for base_col in relative_bases:
            if base_col in race_df.columns:
                vals = pd.to_numeric(race_df[base_col], errors='coerce')
                mean_v = vals.mean(skipna=True)
                std_v = vals.std(ddof=0, skipna=True)
                if pd.isna(mean_v): mean_v = np.nan
                if pd.isna(std_v) or std_v == 0: std_v = 1.0

                race_df[f'{base_col}_race_diff'] = vals - mean_v
                race_df[f'{base_col}_race_zscore'] = (vals - mean_v) / std_v
                race_df[f'{base_col}_race_rank'] = vals.rank(ascending=False, method='min')

        X_pred = pd.DataFrame(index=race_df.index)
        for col in features:
            if col in race_df.columns:
                X_pred[col] = pd.to_numeric(race_df[col], errors='coerce')
            else:
                X_pred[col] = np.nan

        X_pred = X_pred.astype(float)

        preds = []
        for m_key in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat']:
            if m_key in model_data and model_data[m_key] is not None:
                preds.append(model_data[m_key].predict(X_pred))

        if preds:
            race_df['rank_score_raw'] = np.mean(preds, axis=0)
            score_mean = race_df['rank_score_raw'].mean(skipna=True)
            score_std = race_df['rank_score_raw'].std(ddof=0, skipna=True)
            if pd.isna(score_std) or score_std == 0: score_std = 1.0
            race_df['score_disp'] = np.round(((race_df['rank_score_raw'] - score_mean) / score_std) * 10 + 50, 1)
        else:
            race_df['score_disp'] = np.nan
            race_df['rank_score_raw'] = np.nan

        col_kyakushitsu = 'prev1_1c' if 'prev1_1c' in race_df.columns else ('first_corner' if 'first_corner' in race_df.columns else None)
        if col_kyakushitsu and col_kyakushitsu in race_df.columns:
            race_df['脚質'] = race_df[col_kyakushitsu].apply(get_kyakushitsu)
        else:
            race_df['脚質'] = "-"

        race_df = race_df.sort_values(by=['rank_score_raw'], ascending=False, na_position='last').reset_index(drop=True)
        race_df['score_disp'] = race_df['score_disp'].fillna("-")

        # 辞書化（NaNの完全排除）
        records = race_df.to_dict('records')
        clean_records = []
        for rec in records:
            c_rec = {}
            for k, v in rec.items():
                if pd.isna(v) or str(v).strip().lower() in ['nan', 'none', '<na>']:
                    c_rec[k] = "-" if k in ['騎手', '調教師', '脚質', '人気', 'オッズ', 'score_disp', '馬名'] else None
                else:
                    c_rec[k] = v
            clean_records.append(c_rec)

        cache_data[str(rid)] = clean_records

    joblib.dump(cache_data, CACHE_FILE)
    print(f"✨ 完了: 出馬表と過去データを完全に結合し '{CACHE_FILE}' を作成しました。")

if __name__ == "__main__":
    main()