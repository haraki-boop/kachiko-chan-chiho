import os
import sys
import re
import numpy as np
import pandas as pd
import joblib
import warnings

from feature_engineering import generate_features

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

FUTURE_CSV = "future_races_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"
CACHE_FILE = "app_cache_chiho.pkl"

def safe_parse_umaban(val, default=0):
    if pd.isna(val): return default
    try:
        s = str(val).strip()
        m = re.search(r'\d+', s)
        return int(m.group(0)) if m else int(float(s))
    except (ValueError, TypeError):
        return default

def calc_track_bias_series(place_codes, track_cond, kyakus):
    bonuses = np.zeros(len(place_codes), dtype=float)
    is_wet = track_cond in ['重', '不良']
    is_sao = track_cond == '稍重'

    is_nige = (kyakus == '逃')
    is_sen  = (kyakus == '先')

    if is_wet:
        bonuses += np.where(is_nige, 4.0, np.where(is_sen, 2.5, 0.0))
    elif is_sao:
        bonuses += np.where(is_nige, 1.5, np.where(is_sen, 1.0, 0.0))

    p_in = place_codes.isin(['42', '45', '47', '48', '50'])
    if is_wet:
        bonuses += np.where(p_in & is_nige, 2.0, np.where(p_in & is_sen, 1.0, 0.0))
        bonuses += np.where((place_codes == '44') & is_nige, 1.5, np.where((place_codes == '44') & is_sen, 1.0, 0.0))
        bonuses += np.where(place_codes.isin(['30', '54']) & (is_nige | is_sen), 1.5, 0.0)

    return np.round(bonuses, 1)

def main():
    print("\n" + "="*50)
    print(" ⚙️ 勝ち子ちゃん 本番推論キャッシュ生成（当日限定版）")
    print("="*50)

    if not os.path.exists(FUTURE_CSV) or not os.path.exists(MODEL_FILE):
        print(f"❌ 必要なファイル ('{FUTURE_CSV}' または '{MODEL_FILE}') が見つかりません。")
        sys.exit(1)

    # 1. モデルおよびマスタデータの読み込み
    print(f"  [INFO] モデルパッケージ '{MODEL_FILE}' を展開中...")
    model_data = joblib.load(MODEL_FILE)
    features = model_data['features']
    past_j = model_data.get('past_j_stats')
    past_t = model_data.get('past_t_stats')
    past_jp = model_data.get('past_jp_stats')

    # 2. 当日出馬表の読み込み
    print(f"  [INFO] 当日出馬表 '{FUTURE_CSV}' を読み込み中...")
    df_raw = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, low_memory=False)
    if df_raw.empty:
        print("❌ 出馬表データが空です。")
        sys.exit(1)

    # 3. 特徴量エンジニアリング（学習時と全く同一の処理を適用）
    print("  [INFO] 当日出走馬の特徴量を動的生成中（マスタ同期）...")
    df_feat = generate_features(df_raw, is_train=False, past_j_stats=past_j, past_t_stats=past_t, past_jp_stats=past_jp)
    df_clean = df_feat.rename(columns=lambda x: re.sub(r'[^\w\s\-_]', '', str(x)))

    for c in features:
        if c not in df_clean.columns:
            df_clean[c] = 0.0

    X_pred = df_clean[features].astype(float)

    # 4. 3大Rankerアンサンブル推論
    print(f"  [INFO] 3大Ranker推論中... (対象: {len(X_pred)} 頭 / {df_clean['race_id_str'].nunique()} レース)")
    preds = []
    if 'model_rank_lgb' in model_data:
        preds.append(model_data['model_rank_lgb'].predict(X_pred))
    if 'model_rank_xgb' in model_data:
        preds.append(model_data['model_rank_xgb'].predict(X_pred))
    if 'model_rank_cat' in model_data:
        preds.append(model_data['model_rank_cat'].predict(X_pred, thread_count=-1))

    df_clean['rank_score_raw'] = np.nanmean(preds, axis=0)

    # 5. レース内正規化（55〜90pt基準）
    def scale_group(s):
        s_min, s_max = s.min(), s.max()
        if s_max == s_min or pd.isna(s_max):
            return pd.Series(70.0, index=s.index)
        return np.round(55 + (s - s_min) / (s_max - s_min) * 35, 1)

    df_clean['score_disp_base'] = df_clean.groupby('race_id_str')['rank_score_raw'].transform(scale_group)

    if '公式脚質' in df_clean.columns:
        df_clean['脚質_アプリ用'] = df_clean['公式脚質'].apply(lambda k: str(k).strip() if str(k).strip() in ['逃', '先', '差', '追'] else '-')
    else:
        df_clean['脚質_アプリ用'] = '-'

    for cond in ['良', '稍重', '重', '不良']:
        bonuses = calc_track_bias_series(df_clean['place_code_str'], cond, df_clean['脚質_アプリ用'])
        df_clean[f'track_bonus_{cond}'] = bonuses
        df_clean[f'score_{cond}'] = np.minimum(100.0, df_clean['score_disp_base'] + bonuses)

    df_clean['umaban_clean'] = df_clean['馬番'].apply(safe_parse_umaban)

    # 6. アプリ表示用辞書生成
    display_cols = [
        'race_id_str', '枠番', '馬番', 'umaban_clean', '馬名', '騎手', '調教師',
        '斤量', '馬体重', '体重増減', '公式脚質', '脚質_アプリ用',
        'score_disp_base', 'rank_score_raw',
        'score_良', 'score_稍重', 'score_重', 'score_不良'
    ]
    sub_df = df_clean[[c for c in display_cols if c in df_clean.columns]].copy()
    sub_df['sort_key'] = sub_df['score_良'].fillna(0)
    sub_df = sub_df.sort_values(by=['race_id_str', 'sort_key', 'rank_score_raw'], ascending=[True, False, False])

    cache_data = {}
    marks = ["◎", "◯", "▲", "△1", "△2"]

    for rid, group in sub_df.groupby('race_id_str', sort=False):
        records = group.to_dict('records')
        picks = []
        for idx in range(min(5, len(records))):
            h = records[idx]
            picks.append({
                "mark": marks[idx],
                "umaban": int(h.get('umaban_clean', 0)),
                "name": str(h.get('馬名', '')),
                "score": float(h.get('score_良', 0.0))
            })

        valid_picks = [p for p in picks if p['umaban'] > 0]
        bet_lines = {}
        if len(valid_picks) >= 4:
            honmei = valid_picks[0]['umaban']
            taikou = valid_picks[1]['umaban']
            tana = valid_picks[2]['umaban']
            osae1 = valid_picks[3]['umaban']
            osae2 = valid_picks[4]['umaban'] if len(valid_picks) >= 5 else None

            bet_lines["単勝"] = f"{honmei:02d} ({valid_picks[0]['name']})"
            maren_targets = [f"{valid_picks[k]['umaban']:02d}" for k in range(1, len(valid_picks))]
            bet_lines["馬連"] = f"軸 {honmei:02d} － 相手 {', '.join(maren_targets)} (計{len(maren_targets)}点)"
            bet_lines["ワイド"] = f"本線: {honmei:02d}－{taikou:02d}, {honmei:02d}－{tana:02d} / 抑え: {honmei:02d}－{osae1:02d}"
            if osae2:
                bet_lines["3連複F"] = f"1頭目: {honmei:02d}\n2頭目: {taikou:02d}, {tana:02d}\n3頭目: {taikou:02d}, {tana:02d}, {osae1:02d}, {osae2:02d} (計7点)"
            else:
                bet_lines["3連複BOX"] = f"{honmei:02d}, {taikou:02d}, {tana:02d}, {osae1:02d} (計4点)"

        cache_data[str(rid)] = {
            "records": records,
            "bet_recommendations": bet_lines,
            "top_picks": picks
        }

    # gzip圧縮付きで保存
    joblib.dump(cache_data, CACHE_FILE, compress=('gzip', 3))
    file_size_kb = os.path.getsize(CACHE_FILE) / 1024
    print(f"  [SUCCESS] '{CACHE_FILE}' 出力完了 (対象: {len(cache_data)} レース, サイズ: {file_size_kb:.1f} KB)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()