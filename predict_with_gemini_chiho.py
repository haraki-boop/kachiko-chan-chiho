import os
import re
import time
import json
import pandas as pd
import numpy as np
import joblib
import unicodedata
import streamlit as st
from datetime import datetime
from google import genai
from google.genai import types

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

st.set_page_config(page_title="AI予想 勝ち子ちゃん | 3連系特化アンサンブル版", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fcf9f9 !important; color: #333333 !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    p, span, label, div, li, td, th { color: #333333; }
    h1 { font-size: 1.9rem !important; color: #c94a65 !important; font-weight: 800; }
    h2 { font-size: 1.4rem !important; color: #5a3d46 !important; }
    .section-header { font-size: 1.25rem; font-weight: 800; color: #c94a65 !important; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #f2cdd5; padding-bottom: 6px; }
    
    .rec-banner-formation {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: #ffffff !important; padding: 18px 24px; border-radius: 12px; font-size: 1.3rem; font-weight: 900;
        box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3); margin-bottom: 25px; border: 2px solid #d35400;
    }
    
    .bias-box {
        background-color: #e8f4f8; padding: 15px; border-radius: 8px; border: 1px solid #bce0ee; margin-bottom: 15px;
    }

    .table-container { width: 100%; overflow-x: auto; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); background-color: #ffffff; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; }
    .kachi-table thead tr { background: linear-gradient(90deg, #d9788f, #e895a7); color: #ffffff !important; font-weight: bold; }
    .kachi-table th { padding: 8px 10px; text-align: center; border-right: 1px solid rgba(255,255,255,0.2); color: #ffffff !important; font-size: 0.85em; }
    .kachi-table td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #f2eced; color: #5a3d46 !important; font-weight: 500; font-size: 0.9em; }
    .kachi-table tbody tr:hover td { background: #fff5f7; }
    .badge-mark { color: #fff !important; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 0.85em; display: inline-block; min-width: 55px; }
    .badge-honmei { background: linear-gradient(135deg, #ff4757, #ff6b81); }
    .badge-taikou { background: linear-gradient(135deg, #3742fa, #5352ed); }
    .badge-tana   { background: linear-gradient(135deg, #2ed573, #7bed9f); }
    .badge-renka  { background: linear-gradient(135deg, #ffa502, #eccc68); color: #222 !important; }
    .badge-keshi  { background: #e0e0e0; color: #666666 !important; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (3連系特化・完全版)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'baba_status' not in st.session_state: st.session_state['baba_status'] = "良"
if 'bias_multipliers' not in st.session_state: 
    st.session_state['bias_multipliers'] = {"逃": 1.0, "先": 1.0, "差": 1.0, "追": 1.0}

def set_race_id(rid): st.session_state['selected_race_id'] = rid
def reset_bias(): st.session_state['bias_multipliers'] = {"逃": 1.0, "先": 1.0, "差": 1.0, "追": 1.0}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HISTORY_CSV, FUTURE_CSV, ML_TARGET_CSV = "prediction_history_chiho.csv", "future_races_chiho.csv", "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar_ensemble.pkl"

NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

# 🔥 修正①: 長音記号「ー」を消さないように修正（馬名の紐付けバグを解消）
def clean_horse_name(name): 
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･._ ]+', '', s).strip()

def format_weight_display(val):
    if pd.isna(val) or str(val).strip() == "" or str(val).strip() == "-": return "-"
    m = re.search(r'\d{3}(?:\([+-]?\d+\))?', str(val))
    return m.group(0) if m else str(val)

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s = str(val).strip()
    m = re.search(r'(\d{3})(?:\(([-+]?\d+)\))?', s)
    if m: return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0
    return 470.0, 0.0

def parse_rank(x):
    if pd.isna(x): return 5.0
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return 5.0

# 🔥 修正②: 競馬特有の文字列データを純粋な数字に翻訳する専用ルール
def parse_race_time(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if ':' in s:
        parts = s.split(':')
        try: return float(parts[0]) * 60 + float(parts[1])
        except: return np.nan
    try: return float(s)
    except: return np.nan

def parse_margin(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s in ['ハナ', 'クビ', 'アタマ', '同着']: return 0.0
    if s == '大差': return 3.0
    if s in ['取消', '除外', '中止', '落馬', '競止']: return np.nan
    try: return float(s)
    except: return np.nan

def parse_corner(val, return_last=False):
    if pd.isna(val): return np.nan
    s = str(val).translate(str.maketrans('１２３４５６７８９０', '1234567890'))
    s = re.sub(r'[^\d\-]', '', s) # 数字とハイフン以外を除去
    if not s: return np.nan
    parts = s.split('-')
    try:
        return float(parts[-1]) if return_last else float(parts[0])
    except: return np.nan

# 🔥 修正③: Warningを隠すのではなく、dtype=strで安全に読み込む根本治療
def load_csv_safe(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0: 
        return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            if not df.empty: return df
        except Exception: continue
    return pd.DataFrame()

@st.cache_resource
def load_model(): 
    if not os.path.exists(MODEL_FILE): 
        st.error(f"⚠️ 致命的エラー: モデルファイル '{MODEL_FILE}' が見つかりません。")
        return None
    try: 
        return joblib.load(MODEL_FILE)
    except Exception as e: 
        st.error(f"⚠️ 致命的エラー: モデルの読み込みに失敗しました。\n詳細: {e}")
        return None

df_past = load_csv_safe(ML_TARGET_CSV)
df_future = load_csv_safe(FUTURE_CSV)
df_history = load_csv_safe(HISTORY_CSV)

if df_past.empty:
    st.error("⚠️ 過去データ (ml_target_data_chiho.csv) の読み込みに失敗しました。AIが馬の能力を判定できません。")

if not df_future.empty and 'race_id' in df_future.columns:
    df_future['place_code'] = df_future['race_id'].astype(str).str[4:6]
    df_future['place_name'] = df_future['place_code'].map(NAR_PLACES).fillna("地方")
    df_future['r_num'] = pd.to_numeric(df_future['race_id'].astype(str).str[10:12], errors='coerce').fillna(1).astype(int)
    df_future['day_label'] = df_future['date'] if 'date' in df_future.columns else datetime.now().strftime("%Y-%m-%d")

model_data = load_model()

def get_col(df, names):
    for n in names:
        if n in df.columns: return df[n]
    return pd.Series([np.nan]*len(df))

@st.cache_data
def build_past_dicts(df_p):
    jockey_dict, horse_dict, waku_dict, trainer_dict, combo_dict = {}, {}, {}, {}, {}
    if not df_p.empty:
        df_p['馬名_clean'] = df_p['馬名'].astype(str).apply(clean_horse_name)
        
        # 🔥 修正④: 翻訳機を使って文字列を正確に数値化
        pass_col = get_col(df_p, ['通過順位', '通過順', 'コーナー通過順'])
        if pass_col.notna().any():
            df_p['parsed_first_corner'] = pass_col.apply(lambda x: parse_corner(x, False))
            df_p['parsed_last_corner'] = pass_col.apply(lambda x: parse_corner(x, True))
        else:
            df_p['parsed_first_corner'] = pd.to_numeric(get_col(df_p, ['1角', 'first_corner']), errors='coerce')
            df_p['parsed_last_corner'] = pd.to_numeric(get_col(df_p, ['4角', 'last_corner']), errors='coerce')
            
        df_p['parsed_margin'] = get_col(df_p, ['着差', 'time_diff']).apply(parse_margin)
        
        l3f_col = get_col(df_p, ['上り', 'last_3f'])
        df_p['parsed_l3f'] = pd.to_numeric(l3f_col.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

        df_p['騎手_clean'] = get_col(df_p, ['騎手']).astype(str).apply(clean_horse_name)
        trainer_col = get_col(df_p, ['調教師'])
        df_p['trainer_clean'] = trainer_col.astype(str).apply(clean_horse_name)
        df_p['jockey_trainer_combo'] = df_p['騎手_clean'] + "_" + df_p['trainer_clean']

        target_col = get_col(df_p, ['着順', 'target_rank'])
        df_p['target_rank_tmp'] = target_col.apply(parse_rank)
        df_p['target_win'] = (df_p['target_rank_tmp'] == 1.0).astype(int)

        for j, m in df_p.groupby('騎手_clean')['target_win'].mean().items(): jockey_dict[j] = m
        for t, m in df_p.groupby('trainer_clean')['target_win'].mean().items(): trainer_dict[t] = m
        for c, m in df_p.groupby('jockey_trainer_combo')['target_win'].mean().items(): combo_dict[c] = m

        df_p['waku_num_tmp'] = pd.to_numeric(get_col(df_p, ['枠番']), errors='coerce').fillna(0)
        df_p['place_code_tmp'] = get_col(df_p, ['race_id']).astype(str).str[4:6]
        df_p['place_waku_combo'] = df_p['place_code_tmp'] + "_" + df_p['waku_num_tmp'].astype(str)
        for w, m in df_p.groupby('place_waku_combo')['target_win'].mean().items(): waku_dict[w] = m

        baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
        df_p['baba_code'] = get_col(df_p, ['馬場']).map(baba_map).fillna(1)
        df_p['is_bad_baba'] = (df_p['baba_code'] >= 3).astype(int)
        df_p['distance_num'] = pd.to_numeric(get_col(df_p, ['距離', 'distance']), errors='coerce').fillna(1400)
        
        MINAMI_KANTO_CODES = ['42', '43', '44', '45']
        df_p['is_minami_kanto'] = df_p['place_code_tmp'].isin(MINAMI_KANTO_CODES).astype(int)

        date_col = get_col(df_p, ['date', '日付'])
        df_p['date_dt'] = pd.to_datetime(date_col, errors='coerce').fillna(pd.to_datetime('2020-01-01'))
            
        for h, group in df_p.sort_values('date_dt').groupby('馬名_clean'):
            r3 = group.tail(3)
            r5 = group.tail(5)
            f_c = r3['parsed_first_corner'].fillna(8.0).mean()
            l_c = r3['parsed_last_corner'].fillna(f_c).mean()
            l_3f = r3['parsed_l3f'].fillna(39.0).mean()
            t_diff = r3['parsed_margin'].fillna(1.5).mean()
            
            avg_rank_3 = r3['target_rank_tmp'].mean()
            avg_rank_5 = r5['target_rank_tmp'].mean()
            
            dist_dict = group.groupby('distance_num')['target_rank_tmp'].apply(lambda x: x.tail(3).mean()).to_dict()
            place_dict = group.groupby('place_code_tmp')['target_rank_tmp'].apply(lambda x: x.tail(3).mean()).to_dict()
            
            bad_baba_mean = group[group['is_bad_baba'] == 1]['target_rank_tmp'].tail(3).mean()
            if pd.isna(bad_baba_mean): bad_baba_mean = avg_rank_3
            
            last_row = group.iloc[-1]
            prev_is_minami = last_row.get('is_minami_kanto', 0)
            
            last_date = group['date_dt'].max()
            days_since = (pd.Timestamp.now() - last_date).days if not pd.isna(last_date) else 14.0
            
            horse_dict[h] = {
                'first_corner': f_c, 
                'last_corner': l_c, 
                'corner_diff': f_c - l_c, 
                'last_3f': l_3f, 
                'time_diff': t_diff,
                'recent_avg_rank_3': avg_rank_3,
                'recent_avg_rank_5': avg_rank_5,
                'days_since_prev': days_since,
                'horse_career_runs': len(group),
                'prev_is_minami': prev_is_minami,
                'dist_dict': dist_dict,
                'place_dict': place_dict,
                'bad_baba_avg_rank': bad_baba_mean
            }
    return jockey_dict, horse_dict, waku_dict, trainer_dict, combo_dict

jockey_dict, horse_dict, waku_dict, trainer_dict, combo_dict = build_past_dicts(df_past)

def get_kyakushitsu(fc): 
    return "逃" if fc <= 2.0 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

def calculate_race_scores(race_id_target, target_df, baba_status="良", bias_dict=None):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy().reset_index(drop=True)
    if race_df.empty: return None

    race_df['place_code'] = pd.to_numeric(race_df['race_id'].astype(str).str[4:6], errors='coerce').fillna(0.0)
    race_df['distance_num'] = pd.to_numeric(race_df.get('distance'), errors='coerce').fillna(1400)
    race_df['weight_num'] = pd.to_numeric(race_df.get('斤量'), errors='coerce').fillna(54.0)
    race_df['馬番_num'] = pd.to_numeric(race_df.get('馬番'), errors='coerce').fillna(0)
    race_df['waku_num'] = pd.to_numeric(race_df.get('枠番'), errors='coerce').fillna(0)
    race_df['馬名_clean'] = race_df.get('馬名', pd.Series(['']*len(race_df))).astype(str).apply(clean_horse_name)
    race_df['騎手_clean'] = race_df.get('騎手', pd.Series(['']*len(race_df))).astype(str).apply(clean_horse_name)
    
    trainer_col = race_df.get('調教師', race_df['騎手_clean'])
    race_df['trainer_clean'] = trainer_col.astype(str).apply(clean_horse_name)
    race_df['jockey_trainer_combo'] = race_df['騎手_clean'] + "_" + race_df['trainer_clean']

    parsed_w = race_df.get('馬体重', pd.Series(['470']*len(race_df))).apply(parse_weight_info)
    race_df['horse_weight'] = parsed_w.apply(lambda x: x[0])
    race_df['weight_diff'] = parsed_w.apply(lambda x: x[1])

    race_df['first_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('first_corner', 8.0))
    race_df['last_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('last_corner', 8.0))
    race_df['corner_diff'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('corner_diff', 0.0))
    race_df['last_3f'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('last_3f', 39.0))
    race_df['time_diff'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('time_diff', 1.5))
    race_df['recent_avg_rank_3'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank_3', 5.0))
    race_df['recent_avg_rank_5'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank_5', 5.0))
    race_df['days_since_prev'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('days_since_prev', 14.0))
    race_df['horse_career_runs'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('horse_career_runs', 5.0))
    race_df['prev_is_minami'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_is_minami', 0))

    def get_same_dist(row):
        h = row['馬名_clean']
        d = row['distance_num']
        return horse_dict.get(h, {}).get('dist_dict', {}).get(d, horse_dict.get(h, {}).get('recent_avg_rank_3', 5.0))

    def get_same_place(row):
        h = row['馬名_clean']
        p = str(row['place_code']).zfill(2)
        return horse_dict.get(h, {}).get('place_dict', {}).get(p, horse_dict.get(h, {}).get('recent_avg_rank_3', 5.0))

    race_df['same_dist_avg_rank'] = race_df.apply(get_same_dist, axis=1)
    race_df['same_place_avg_rank'] = race_df.apply(get_same_place, axis=1)
    race_df['bad_baba_avg_rank'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('bad_baba_avg_rank', 5.0))

    race_df['斤量'] = race_df['weight_num']
    race_df['jockey_win_rate'] = race_df['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
    race_df['trainer_win_rate'] = race_df['trainer_clean'].apply(lambda x: trainer_dict.get(x, 0.05))
    race_df['combo_win_rate'] = race_df['jockey_trainer_combo'].apply(lambda x: combo_dict.get(x, 0.05))
    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)
    
    MINAMI_KANTO_CODES = ['42', '43', '44', '45']
    race_df['place_code_str'] = race_df['race_id'].astype(str).str[4:6]
    race_df['is_minami_kanto'] = race_df['place_code_str'].isin(MINAMI_KANTO_CODES).astype(int)

    race_df['is_front_runner'] = (race_df['first_corner'] <= 3.0).astype(int)
    race_df['race_front_runners'] = race_df['is_front_runner'].sum()
    
    race_df['place_waku_combo'] = race_df['place_code_str'] + "_" + race_df['waku_num'].astype(str)
    race_df['waku_win_rate'] = race_df['place_waku_combo'].apply(lambda x: waku_dict.get(x, 0.05))

    race_df['prev_1c'] = race_df['first_corner']
    race_df['last_3f_avg_rank'] = race_df['last_3f']
    race_df['avg_time_diff'] = race_df['time_diff']
    race_df['body_weight'] = race_df['horse_weight']
    race_df['kinryo_weight_ratio'] = race_df['斤量'] / race_df['body_weight'].clip(lower=350.0)
    race_df['is_large_weight_change'] = (race_df['weight_diff'].abs() >= 10.0).astype(int)
    
    baba_map = {'良': 1, '稍重': 2, '重': 3, '不良': 4}
    race_df['is_bad_baba'] = 1 if baba_map.get(baba_status, 1) >= 3 else 0

    race_df['custom_time_index'] = 75.0 - (race_df['recent_avg_rank_3'] - 3.0) * 3.5 + (race_df['斤量'] - 54.0) * 1.5
    race_df['custom_start_index'] = (12.0 - race_df['prev_1c'].clip(upper=10.0)) * 6.5

    race_df['place_prob'] = 0.0
    race_df['win_prob'] = 0.0
    
    if model_data and isinstance(model_data, dict):
        try:
            m_feat = model_data.get('features', [])
            m_place_lgb = model_data.get('model_place_lgb')
            m_win_lgb = model_data.get('model_win_lgb')
            m_place_xgb = model_data.get('model_place_xgb')
            m_win_xgb = model_data.get('model_win_xgb')
            m_place_cat = model_data.get('model_place_cat')
            m_win_cat = model_data.get('model_win_cat')
            
            if m_feat and m_place_lgb:
                X = race_df.copy()
                for f in m_feat:
                    if f not in X.columns: X[f] = 0.0
                
                # エラー回避のため確実にfloat化
                X_input = X[m_feat].fillna(0.0).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
                
                p_lgb = m_place_lgb.predict_proba(X_input)[:, 1]
                w_lgb = m_win_lgb.predict_proba(X_input)[:, 1]
                p_xgb = m_place_xgb.predict_proba(X_input)[:, 1]
                w_xgb = m_win_xgb.predict_proba(X_input)[:, 1]
                p_cat = m_place_cat.predict_proba(X_input)[:, 1]
                w_cat = m_win_cat.predict_proba(X_input)[:, 1]
                
                race_df['place_prob'] = (p_lgb + p_xgb + p_cat) / 3.0
                race_df['win_prob'] = (w_lgb + w_xgb + w_cat) / 3.0
                
                sum_win = race_df['win_prob'].sum()
                if sum_win > 0:
                    race_df['win_prob'] = race_df['win_prob'] / sum_win
                else:
                    race_df['win_prob'] = 1.0 / len(race_df)
                    
                sum_place = race_df['place_prob'].sum()
                if sum_place > 0:
                    race_df['place_prob'] = (race_df['place_prob'] / sum_place) * 2.0
                    race_df['place_prob'] = race_df['place_prob'].clip(upper=1.0)
                else:
                    race_df['place_prob'] = min(1.0, 2.0 / len(race_df))
                    
        except Exception as e: 
            st.error(f"⚠️ 推論実行中にエラーが発生しました。\n詳細: {e}")
            st.stop()
    else:
        st.error("⚠️ エラー: AIモデルがロードされていないため、スコア計算を停止します。")
        st.stop()

    w_max, w_min = race_df['win_prob'].max(), race_df['win_prob'].min()
    p_max, p_min = race_df['place_prob'].max(), race_df['place_prob'].min()
    
    if (w_max - w_min) > 1e-4:
        race_df['win_norm'] = (race_df['win_prob'] - w_min) / (w_max - w_min)
    else:
        race_df['win_norm'] = 0.5
        
    if (p_max - p_min) > 1e-4:
        race_df['rentai_norm'] = (race_df['place_prob'] - p_min) / (p_max - p_min)
    else:
        race_df['rentai_norm'] = 0.5
    
    race_df['raw_score'] = (race_df['win_norm'] * 0.60) + (race_df['rentai_norm'] * 0.30)
    
    if bias_dict:
        race_df['bias_multiplier'] = race_df['脚質'].map(bias_dict).fillna(1.0)
        race_df['raw_score'] = race_df['raw_score'] * race_df['bias_multiplier']

    r_max, r_min = race_df['raw_score'].max(), race_df['raw_score'].min()
    if (r_max - r_min) > 1e-4:
        race_df['score_disp'] = (((race_df['raw_score'] - r_min) / (r_max - r_min)) * 100).astype(int)
    else:
        race_df['score_disp'] = 50

    return race_df.sort_values(by=['raw_score'], ascending=False).reset_index(drop=True)

st.sidebar.header("🔄 画面の更新")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
if st.sidebar.button("🔄 キャッシュ完全クリア＆リロード", use_container_width=True): 
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

def get_mark(idx):
    if idx == 0: return "◎ 本命"
    elif idx == 1: return "◯ 対抗"
    elif idx == 2: return "▲ 単穴"
    elif idx == 3: return "△ 連下"
    elif idx == 4: return "☆ 穴馬"
    else: return "消"

def generate_beautiful_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>馬体重</th><th>騎手</th><th>脚質</th><th>AIスコア</th><th>1着率</th><th>連対率</th><th>印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        mark = get_mark(i)
        b_cls = "badge-honmei" if "◎" in mark else "badge-taikou" if "◯" in mark else "badge-tana" if "▲" in mark else "badge-renka" if "△" in mark else "badge-tana" if "☆" in mark else "badge-keshi"
        kyaku = r.get('脚質', '-')
        k_style = "background:#ff7675;" if kyaku == "逃" else "background:#e67e22;" if kyaku == "先" else "background:#3498db;" if kyaku == "差" else "background:#2ecc71;"

        actual_w = str(r.get('馬体重', '-')).strip()
        if actual_w in ["", "-", "nan", "NaN", "None"]:
            body_w = int(r.get('horse_weight', 470))
            weight_str = f"<span style='color:#aaa; font-size:0.9em;'>{body_w}<br>(前走)</span>"
        else:
            weight_str = f"<b>{format_weight_display(actual_w)}</b>"
        
        html += f"""<tr>
<td style='font-weight:bold; color:#c94a65 !important;'>{int(r['馬番_num']):02d}</td>
<td style='text-align:left; font-weight:800; color:#5a3d46 !important;'>{r.get('馬名', '-')}</td>
<td>{weight_str}</td>
<td style='color:#666666 !important;'>{r.get('騎手', '-')}</td>
<td><span style='{k_style} color:#fff !important; padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{kyaku}</span></td>
<td style='color:#5a3d46 !important;'><b>{int(r['score_disp'])}点</b></td>
<td style='color:#c94a65 !important; font-size:1.05em;'><b>{r['win_prob']*100:.1f}%</b></td>
<td style='color:#666666 !important;'>{r['place_prob']*100:.1f}%</td>
<td><span class='badge-mark {b_cls}'>{mark}</span></td>
</tr>"""
    html += "</tbody></table></div>"
    return html

tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想 (完全版)", "📈 地方実戦成績"])

with tab_forecast:
    if df_future.empty: st.warning("⚠️ 出馬表データが存在しません。")
    else:
        st.markdown("<div class='section-header'>🎯 予想レースを選択</div>", unsafe_allow_html=True)
        sel_date = st.radio("開催日", sorted(df_future['day_label'].unique()), horizontal=True, label_visibility="collapsed")
        day_df = df_future[df_future['day_label'] == sel_date]
        places = day_df['place_name'].unique()
        
        place_tabs = st.tabs([f"📍 {p}" for p in places])
        for p_idx, place in enumerate(places):
            with place_tabs[p_idx]:
                place_df = day_df[day_df['place_name'] == place]
                for i in range(0, len(place_df['r_num'].unique()), 6):
                    cols = st.columns(6)
                    for j, r in enumerate(sorted(place_df['r_num'].unique())[i:i+6]):
                        rid = place_df[place_df['r_num'] == r]['race_id'].iloc[0]
                        cols[j].button(
                            f"{r}R", 
                            key=f"btn_{rid}", 
                            use_container_width=True, 
                            type="primary" if st.session_state['selected_race_id'] == rid else "secondary",
                            on_click=set_race_id, args=(rid,)
                        )

    if st.session_state['selected_race_id'] and not df_future.empty:
        st.markdown("---")
        st.markdown("<div class='section-header'>🌤️ リアルタイム馬場バイアス補正 (Gemini AI)</div>", unsafe_allow_html=True)
        
        b_cols1, b_cols2 = st.columns([3, 1])
        with b_cols1:
            user_bias_text = st.text_input("今日の馬場傾向を入力（例: 「今日は前残りが凄い」「差しが決まる」など）")
        with b_cols2:
            st.write("")
            st.write("")
            if st.button("🧠 Geminiでバイアス係数を算出", use_container_width=True):
                if not api_key_input:
                    st.error("APIキーを入力してください。")
                elif not user_bias_text:
                    st.warning("馬場傾向のテキストを入力してください。")
                else:
                    with st.spinner("Geminiが馬場傾向を解析中..."):
                        try:
                            sys_inst = """ユーザーが入力した競馬の馬場傾向テキストを解析し、各脚質（逃げ・先行・差し・追込）の勝率に対する補正倍率（0.8〜1.5の範囲の数値）を出力してください。
必ず以下のJSON形式のみを出力してください。Markdownタグ(```json)などは一切含めないでください。
{"逃": 1.2, "先": 1.1, "差": 0.9, "追": 0.8}
"""
                            ai_client = genai.Client(api_key=api_key_input)
                            response = ai_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=user_bias_text,
                                config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.1)
                            )
                            match = re.search(r'\{.*?\}', response.text, re.DOTALL)
                            if match:
                                parsed_dict = json.loads(match.group(0))
                                st.session_state['bias_multipliers'] = {
                                    k: float(v) for k, v in parsed_dict.items() if k in ["逃", "先", "差", "追"]
                                }
                                st.success("✅ 馬場バイアスをAIスコアに反映しました！")
                            else:
                                st.error("JSONデータのパースに失敗しました。")
                        except Exception as e:
                            st.error(f"エラーが発生しました: {e}")

        bm = st.session_state['bias_multipliers']
        st.markdown(f"""
        <div class='bias-box'>
            <b>現在適用中のバイアス倍率:</b> 
            <span style='margin-left:10px;'>🏃 逃げ: <b>{bm.get('逃', 1.0):.1f}倍</b></span> | 
            <span>🏇 先行: <b>{bm.get('先', 1.0):.1f}倍</b></span> | 
            <span>🐎 差し: <b>{bm.get('差', 1.0):.1f}倍</b></span> | 
            <span>🌪️ 追込: <b>{bm.get('追', 1.0):.1f}倍</b></span>
            <br><span style='font-size:0.85em; color:#666;'>※入力内容に基づき、AIの生スコアに上記の倍率が掛け合わされています。</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 バイアスをリセット (1.0倍に戻す)"):
            reset_bias()
            st.rerun()

        st.markdown("---")
        target_id = str(st.session_state['selected_race_id'])
        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0]
        race_display_name = f"{info['place_name']} {info['r_num']}R 【{info.get('race_name', '')}】"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        cond = st.radio("🌧️ 現在の馬場状態を選択してください", ["良", "稍重", "重", "不良"], horizontal=True, index=["良", "稍重", "重", "不良"].index(st.session_state['baba_status']))
        if cond != st.session_state['baba_status']: 
            st.session_state['baba_status'] = cond
            st.rerun()
        
        scored_df = calculate_race_scores(target_id, df_future, baba_status=st.session_state['baba_status'], bias_dict=st.session_state['bias_multipliers'])
        
        if scored_df is not None:
            def safe_idx(df, idx):
                return int(df.iloc[idx]['馬番_num']) if len(df) > idx else int(df.iloc[-1]['馬番_num'])
            
            u_1 = safe_idx(scored_df, 0)
            u_2 = safe_idx(scored_df, 1)
            u_3 = safe_idx(scored_df, 2)
            u_4 = safe_idx(scored_df, 3)
            u_5 = safe_idx(scored_df, 4)

            score_diff = scored_df.iloc[0]['score_disp'] - scored_df.iloc[1]['score_disp'] if len(scored_df) > 1 else 10
            
            front_runners_count = int(scored_df.iloc[0]['race_front_runners']) if 'race_front_runners' in scored_df.columns else 0
            pace_text = f"<br>🔥 <b>展開予想:</b> このレースは逃げ・先行馬が {front_runners_count} 頭います。{'ハイペース崩れに注意！差し馬の評価を上げています。' if front_runners_count >= 3 else 'ペースは落ち着きそうです。前残り注意。'}"

            if score_diff >= 5:
                rec_pattern_name = "🎯 【3連単・1着固定流し】 1位 ➔ 2〜4位 (計6点)"
                rec_text = f"1位のスコアが抜けている（{score_diff}点差）ため、頭固定の3連単で高配当を狙い撃ちします。"
                axis_horse = f"{u_1:02d}"
                target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}"
            else:
                rec_pattern_name = "🛡️ 【3連複・1頭軸流し】 1位 ➔ 2〜5位 (計6点)"
                rec_text = f"上位陣が混戦（{score_diff}点差）なため、1位を軸にしつつ相手を5位まで広げた3連複で高配当を狙います。"
                axis_horse = f"{u_1:02d}"
                target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}, {u_5:02d}"

            st.markdown(f"""
            <div class='rec-banner-formation'>
                {rec_pattern_name}<br>
                <span style='font-size:0.85em; font-weight:normal;'>
                * <b>軸馬(1頭):</b> <b>{axis_horse}</b><br>
                * <b>相手(ヒモ):</b> {target_horses}<br>
                * <b>理由:</b> 🤖 <b>【LightGBM × XGBoost × CatBoost】3大AI合議スコア</b>に基づく3連系特化ロジックです。{rec_text}{pace_text}
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAIスコア (📍 第3形態・3AI合議アンサンブル版)</div>", unsafe_allow_html=True)
            st.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Geminiの見解（解説テキスト）を生成する", use_container_width=True):
            if not api_key_input: 
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            table_summary = []
            for idx, row in scored_df.head(5).iterrows():
                mark = get_mark(idx)
                if mark != "消":
                    table_summary.append(
                        f"印:{mark} | 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | 脚質:{row['脚質']} | AIスコア:{row['score_disp']} | 1着率:{row['win_prob']*100:.1f}%"
                    )

            sys_inst = f"""あなたは地方競馬の勝ち子ちゃんです。
Markdownの見出しタグ（###や---など）は使わず、絵文字混じりの綺麗な文章で回答してください。

競馬場: {info['place_name']} / 馬場: {st.session_state['baba_status']}
適用中のバイアス: 逃げ {bm.get('逃')}倍, 先行 {bm.get('先')}倍, 差し {bm.get('差')}倍, 追込 {bm.get('追')}倍
自動判定された買い目戦略: {rec_pattern_name}

【回答の構成】
🌸 勝ち子ちゃんの展開の見解
（3大AIの合議判定とトラックバイアス係数をどのように反映したかを含めて短評を入れる）

🎯 注目馬解説 (3連単・3連複の観点で)
◎ 本命: 馬番・馬名（理由）
◯ 対抗: 馬番・馬名（理由）
▲ 単穴: 馬番・馬名（理由）
△ 連下: 馬番・馬名（理由）
☆ 穴馬: 馬番・馬名（理由）
"""
            with st.spinner("🎀 分析の見解を作成中..."):
                try:
                    ai_client = genai.Client(api_key=api_key_input)
                    response = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"対象レース: {race_display_name}\n\n対象馬:\n" + "\n".join(table_summary),
                        config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.3)
                    )
                    clean_text = re.sub(r'^[#\-\s]+', '', response.text.strip())
                    st.markdown(f"<div class='gemini-output-box'>{clean_text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"エラー: {e}")

with tab_dashboard:
    st.markdown("<div class='section-header'>📈 地方実戦成績ダッシュボード</div>", unsafe_allow_html=True)
    if df_history.empty: st.info("まだ履歴がありません。")
    else: st.info("ダッシュボードは稼働中です。")