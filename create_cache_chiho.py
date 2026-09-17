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
    if pd.isna(name_val): return ""
    return re.sub(r'[\s・･._\u3000\t\r\n]+', '', unicodedata.normalize('NFKC', str(name_val))).strip().upper()

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

    # 1. 最新過去データのロード
    df_past = pd.read_csv(CSV_PAST, low_memory=False) if os.path.exists(CSV_PAST) else pd.DataFrame()
    
    if not df_past.empty:
        horse_col = '馬名_clean' if '馬名_clean' in df_past.columns else ('馬名' if '馬名' in df_past.columns else 'horse_name')
        df_past['馬名_clean'] = df_past[horse_col].astype(str).apply(clean_horse_name)
        df_past_latest = df_past.groupby('馬名_clean').last().reset_index()
    else:
        df_past_latest = pd.DataFrame()

    # 2. 出馬表ロード
    df_future = pd.read_csv(CSV_FUTURE, low_memory=False)
    df_future['馬名_clean'] = df_future['馬名'].astype(str).apply(clean_horse_name)
    df_future['騎手_clean'] = df_future['騎手'].astype(str).str.strip()
    df_future['調教師_clean'] = df_future.get('調教師', pd.Series("")).astype(str).str.strip()
    df_future['race_id_clean'] = pd.to_numeric(df_future['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)

    # 3. 過去最新データと出馬表のマージ
    if not df_past_latest.empty:
        cols_to_drop = [c for c in df_past_latest.columns if c in df_future.columns and c != '馬名_clean']
        df_past_clean = df_past_latest.drop(columns=cols_to_drop)
        df_merged = pd.merge(df_future, df_past_clean, on='馬名_clean', how='left')
    else:
        df_merged = df_future.copy()

    # 4. レース単位での動的特徴量算出 (Diff, Zscore, Rank)
    cache_data = {}
    races = df_merged['race_id_clean'].unique()

    relative_bases = set()
    for f in features:
        if f.endswith('_race_diff'):
            relative_bases.add(f[:-10])
        elif f.endswith('_race_zscore'):
            relative_bases.add(f[:-12])
        elif f.endswith('_race_rank'):
            relative_bases.add(f[:-10])

    for rid in races:
        if str(rid) == '0': continue
        race_df = df_merged[df_merged['race_id_clean'] == rid].copy()
        if race_df.empty: continue

        # レース内相対評価の算出（欠損はNaNのまま）
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

        # モデル入力行列 (X_pred) の作成
        X_pred = pd.DataFrame(index=race_df.index)
        for col in features:
            if col in race_df.columns:
                X_pred[col] = pd.to_numeric(race_df[col], errors='coerce')
            else:
                X_pred[col] = np.nan

        X_pred = X_pred.astype(float)

        # アンサンブル推論
        preds = []
        for m_key in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat']:
            if m_key in model_data and model_data[m_key] is not None:
                preds.append(model_data[m_key].predict(X_pred))

        if preds:
            race_df['rank_score_raw'] = np.mean(preds, axis=0)
            score_mean = race_df['rank_score_raw'].mean(skipna=True)
            score_std = race_df['rank_score_raw'].std(ddof=0, skipna=True)
            if pd.isna(score_std) or score_std == 0: score_std = 1.0
            race_df['score_disp'] = np.round(((race_df['rank_score_raw'] - score_mean) / score_std) * 10 + 50).astype(object)
        else:
            race_df['score_disp'] = "-"
            race_df['rank_score_raw'] = np.nan

        # 脚質表示の判定
        col_kyakushitsu = 'prev1_1c' if 'prev1_1c' in race_df.columns else ('first_corner' if 'first_corner' in race_df.columns else None)
        if col_kyakushitsu and col_kyakushitsu in race_df.columns:
            race_df['脚質'] = race_df[col_kyakushitsu].apply(get_kyakushitsu)
        else:
            race_df['脚質'] = "-"

        # スコアがない（NaN）の馬は最下位になるようソート
        race_df = race_df.sort_values(by=['rank_score_raw'], ascending=False, na_position='last').reset_index(drop=True)
        
        # 表示用テキスト補完（数値系NaNは"-"表示化）
        race_df['score_disp'] = race_df['score_disp'].fillna("-")
        for col in race_df.columns:
            if race_df[col].dtype == object:
                race_df[col] = race_df[col].fillna("")

        cache_data[str(rid)] = race_df.to_dict('records')

    joblib.dump(cache_data, CACHE_FILE)
    print(f"✨ 完了: '{CACHE_FILE}' を作成しました。(データ無し馬の50補完を完全排除)")

if __name__ == "__main__":
    main()