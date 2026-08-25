import os
import re
import time
import pandas as pd
import numpy as np
import joblib
import unicodedata
import streamlit as st
from datetime import datetime
from google import genai
from google.genai import types

# ==========================================
# 🌸 1. カスタムCSS & デザイン設定
# ==========================================
st.set_page_config(page_title="AI予想 勝ち子ちゃん | 軸1頭相手4頭流し(6点)高的中モデル", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fcf9f9 !important; color: #333333 !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    p, span, label, div, li, td, th { color: #333333; }
    h1 { font-size: 1.9rem !important; color: #c94a65 !important; font-weight: 800; }
    h2 { font-size: 1.4rem !important; color: #5a3d46 !important; }
    .section-header { font-size: 1.25rem; font-weight: 800; color: #c94a65 !important; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #f2cdd5; padding-bottom: 6px; }
    
    /* 🚨 最上部ハイライトバナーデザイン */
    .rec-banner-through {
        background: linear-gradient(135deg, #ff4d4d, #ff3333);
        color: #ffffff !important;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 900;
        box-shadow: 0 4px 15px rgba(255, 77, 77, 0.3);
        margin-bottom: 25px;
        border: 2px solid #cc0000;
    }
    .rec-banner-one-strong {
        background: linear-gradient(135deg, #00b894, #00bec4);
        color: #ffffff !important;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 900;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
        margin-bottom: 25px;
        border: 2px solid #008060;
    }
    .rec-banner-top4 {
        background: linear-gradient(135deg, #ff9f43, #feca57);
        color: #222222 !important;
        padding: 18px 24px;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 900;
        box-shadow: 0 4px 15px rgba(255, 159, 67, 0.3);
        margin-bottom: 25px;
        border: 2px solid #e67e22;
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
    .badge-ana    { background: linear-gradient(135deg, #a55eea, #4b7bec); }
    .badge-keshi  { background: #e0e0e0; color: #666666 !important; }
    .badge-alert { background: linear-gradient(135deg, #e1b12c, #fbc531); color: #000 !important; font-weight: 900; padding: 3px 8px; border-radius: 6px; }
    .gemini-output-box { background-color: #ffffff !important; color: #222222 !important; padding: 20px; border-radius: 12px; border: 2px solid #f2cdd5; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (軸1頭相手4頭流し 6点勝負モデル)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'baba_status' not in st.session_state: st.session_state['baba_status'] = "良"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HISTORY_CSV, FUTURE_CSV, ML_TARGET_CSV, MODEL_FILE = "prediction_history_chiho.csv", "future_races_chiho.csv", "ml_target_data_chiho.csv", "keiba_ai_model_nar.pkl"
NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

def clean_horse_name(name): 
    return re.sub(r'[\s\u3000]+', '', unicodedata.normalize('NFKC', str(name))) if not pd.isna(name) else ""

def format_weight_display(val):
    if pd.isna(val) or str(val).strip() == "" or str(val).strip() == "-":
        return "-"
    m = re.search(r'\d{3}(?:\([+-]?\d+\))?', str(val))
    return m.group(0) if m else str(val)

def parse_sex_age(val):
    if pd.isna(val): return 0, 4.0
    s = str(val).strip()
    sex_code = 1 if '牝' in s else (2 if 'セ' in s else 0)
    age_match = re.search(r'\d+', s)
    age = float(age_match.group(0)) if age_match else 4.0
    return sex_code, age

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s = str(val).strip()
    m = re.search(r'(\d{3})(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return 470.0, 0.0

def load_csv_safe(path, dtype_dict=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
        try:
            df = pd.read_csv(path, dtype=dtype_dict, encoding=enc)
            if not df.empty: return df
        except Exception: continue
    return pd.DataFrame()

@st.cache_resource
def load_model(): 
    if not os.path.exists(MODEL_FILE): return None
    try: return joblib.load(MODEL_FILE)
    except Exception: return None

df_past = load_csv_safe(ML_TARGET_CSV, {'race_id': str})
df_future = load_csv_safe(FUTURE_CSV, {'race_id': str, '馬番': str})
df_history = load_csv_safe(HISTORY_CSV, {'race_id': str})

if not df_future.empty and 'race_id' in df_future.columns:
    df_future['place_code'] = df_future['race_id'].astype(str).str[4:6]
    df_future['place_name'] = df_future['place_code'].map(NAR_PLACES).fillna("地方")
    df_future['r_num'] = df_future['race_id'].astype(str).str[10:12].astype(int)
    df_future['day_label'] = df_future['date'] if 'date' in df_future.columns else datetime.now().strftime("%Y-%m-%d")

model_data = load_model()

@st.cache_data
def build_past_dicts(df_p):
    jockey_dict, trainer_dict, combo_dict, owner_dict, horse_dict, waku_place_dict = {}, {}, {}, {}, {}, {}
    if not df_p.empty:
        df_p['馬名_clean'] = df_p['馬名'].astype(str).apply(clean_horse_name)
        df_p['first_corner'] = pd.to_numeric(df_p.get('first_corner', df_p.get('1角')), errors='coerce').fillna(8.0)
        df_p['last_corner'] = pd.to_numeric(df_p.get('last_corner', df_p.get('4角')), errors='coerce').fillna(df_p['first_corner'])
        df_p['last_3f'] = pd.to_numeric(df_p.get('last_3f', df_p.get('上り')), errors='coerce')
        df_p['time_diff'] = pd.to_numeric(df_p.get('time_diff', df_p.get('着差')), errors='coerce').fillna(1.5)
        df_p['waku_num'] = pd.to_numeric(df_p.get('枠番'), errors='coerce').fillna(0)
        df_p['distance_num'] = pd.to_numeric(df_p.get('distance'), errors='coerce').fillna(1400)
        
        MINAMI_CODES = ['42', '43', '44', '45']
        df_p['place_code'] = df_p['race_id'].astype(str).str[4:6]
        df_p['is_minami'] = df_p['place_code'].isin(MINAMI_CODES).astype(int)
        
        baba_map = {'良': 1, '稍': 2, '稍重': 2, '重': 3, '不': 4, '不良': 4}
        df_p['baba_code'] = df_p.get('馬場', pd.Series(['良']*len(df_p))).map(baba_map).fillna(1)
        df_p['is_bad_baba'] = (df_p['baba_code'] >= 3).astype(int)

        trainer_col = '調教師' if '調教師' in df_p.columns else '騎手'
        df_p['trainer_clean'] = df_p[trainer_col].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip() if pd.notna(x) else '')
        df_p['騎手_clean'] = df_p['騎手'].astype(str).apply(clean_horse_name)
        df_p['owner_clean'] = df_p['馬主'].astype(str).apply(clean_horse_name) if '馬主' in df_p.columns else ''
        df_p['combo_key'] = df_p['騎手_clean'] + "_" + df_p['trainer_clean']

        if 'target_win' in df_p.columns:
            for j, m in df_p.groupby('騎手_clean')['target_win'].mean().items(): jockey_dict[j] = m
            for t, m in df_p.groupby('trainer_clean')['target_win'].mean().items(): trainer_dict[t] = m
            for c, m in df_p.groupby('combo_key')['target_win'].mean().items(): combo_dict[c] = m
            for o, m in df_p.groupby('owner_clean')['target_win'].mean().items(): owner_dict[o] = m
            for wp, m in df_p.groupby(['place_code', 'distance_num', 'waku_num'])['target_win'].mean().items(): waku_place_dict[wp] = m

        df_p['date_dt'] = pd.to_datetime(df_p.get('date', pd.Series(['2020-01-01']*len(df_p))), errors='coerce')

        for h, group in df_p.sort_values('date_dt').groupby('馬名_clean'):
            r3, r5 = group.tail(3), group.tail(5)
            last_row = r3.iloc[-1] if len(r3) > 0 else {}
            last_date = last_row.get('date_dt', pd.Timestamp('2020-01-01'))
            days_diff = (datetime.now() - last_date).days if pd.notna(last_date) else 30.0

            bad_group = group[group['is_bad_baba'] == 1]
            bad_avg = bad_group.tail(3)['target_rank'].mean() if len(bad_group) > 0 else r3.get('target_rank', pd.Series([6.0])).mean()
            
            f_c = pd.to_numeric(r3.get('first_corner', pd.Series([8.0])), errors='coerce').fillna(8.0).mean()
            l_c = pd.to_numeric(r3.get('last_corner', pd.Series([8.0])), errors='coerce').fillna(f_c).mean()
            
            horse_dict[h] = {
                'prev_dist': last_row.get('distance_num', 1400),
                'first_corner': f_c,
                'last_corner': l_c,
                'corner_diff': f_c - l_c,
                'last_3f_avg_rank': pd.to_numeric(r3.get('last_3f', pd.Series([39.0])), errors='coerce').fillna(39.0).mean(),
                'avg_time_diff': pd.to_numeric(r3.get('time_diff', pd.Series([1.5])), errors='coerce').fillna(1.5).mean(),
                'recent_avg_rank_3': r3.get('target_rank', pd.Series([6.0])).mean(),
                'recent_avg_rank_5': r5.get('target_rank', pd.Series([6.0])).mean(),
                'bad_baba_avg_rank': bad_avg,
                'career_runs': len(group),
                'prev_is_minami': last_row.get('is_minami', 0),
                'days_since_prev': days_diff,
                'same_place_avg_rank': group.groupby('place_code')['target_rank'].mean().to_dict(),
                'mid_speed': pd.to_numeric(r3.get('mid_speed', pd.Series([1500.0])), errors='coerce').fillna(1500.0).mean(),
                'custom_pursuit_index': pd.to_numeric(r3.get('custom_pursuit_index', pd.Series([50.0])), errors='coerce').fillna(50.0).mean()
            }
    return jockey_dict, trainer_dict, combo_dict, owner_dict, horse_dict, waku_place_dict

jockey_dict, trainer_dict, combo_dict, owner_dict, horse_dict, waku_place_dict = build_past_dicts(df_past)

def get_kyakushitsu(fc): 
    return "逃" if fc <= 2.0 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

def calculate_race_scores(race_id_target, target_df, baba_status="良"):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy().reset_index(drop=True)
    if race_df.empty: return None

    MINAMI_CODES = ['42', '43', '44', '45']
    race_df['place_code'] = race_df['race_id'].astype(str).str[4:6]
    race_df['is_minami_kanto'] = race_df['place_code'].isin(MINAMI_CODES).astype(int)

    baba_map = {'良': 1, '稍重': 2, '重': 3, '不良': 4}
    race_df['baba_code'] = baba_map.get(baba_status, 1)
    race_df['is_bad_baba'] = (race_df['baba_code'] >= 3).astype(int)

    race_df['単勝_num'] = pd.to_numeric(race_df['単勝'], errors='coerce').fillna(15.0) if '単勝' in race_df.columns else 15.0
    race_df['distance_num'] = pd.to_numeric(race_df['distance'], errors='coerce').fillna(1400) if 'distance' in race_df.columns else 1400
    race_df['斤量'] = pd.to_numeric(race_df['斤量'], errors='coerce').fillna(54.0) if '斤量' in race_df.columns else 54.0
    race_df.loc[(race_df['斤量'] == race_df['単勝_num']) | (race_df['斤量'] < 48.0) | (race_df['斤量'] > 63.0), '斤量'] = 54.0
    race_df['馬番_num'] = pd.to_numeric(race_df['馬番'], errors='coerce').fillna(0) if '馬番' in race_df.columns else 0
    race_df['waku_num'] = pd.to_numeric(race_df['枠番'], errors='coerce').fillna(0) if '枠番' in race_df.columns else 0
    race_df['馬名_clean'] = race_df['馬名'].astype(str).apply(clean_horse_name)

    if '性齢' in race_df.columns:
        parsed_sa = race_df['性齢'].apply(parse_sex_age)
        race_df['sex_code'], race_df['age'] = [x[0] for x in parsed_sa], [x[1] for x in parsed_sa]
    else: race_df['sex_code'], race_df['age'] = 0, 4.0

    if '馬体重' in race_df.columns:
        parsed_w = race_df['馬体重'].apply(parse_weight_info)
        race_df['body_weight'], race_df['body_weight_diff'] = [x[0] for x in parsed_w], [x[1] for x in parsed_w]
    else: 
        race_df['body_weight'], race_df['body_weight_diff'] = 470.0, 0.0

    race_df.loc[race_df['body_weight'] == 470.0, 'body_weight'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_weight', 470.0))
    race_df['kinryo_weight_ratio'] = race_df['斤量'] / race_df['body_weight'].clip(lower=350.0)
    race_df['is_large_weight_change'] = (race_df['body_weight_diff'].abs() >= 10.0).astype(int)

    race_df['first_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('first_corner', 6.0))
    race_df['last_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('last_corner', 6.0))
    race_df['corner_diff'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('corner_diff', 0.0))
    race_df['prev_1c'] = race_df['first_corner']

    race_df['last_3f_avg_rank'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('last_3f_avg_rank', 39.0))
    race_df['avg_time_diff'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('avg_time_diff', 1.5))
    race_df['recent_avg_rank_3'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank_3', 6.0))
    race_df['recent_avg_rank_5'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank_5', 6.0))
    race_df['bad_baba_avg_rank'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('bad_baba_avg_rank', 6.0))
    race_df['same_dist_avg_rank'] = race_df['recent_avg_rank_3']
    race_df['same_place_avg_rank'] = race_df.apply(lambda r: horse_dict.get(r['馬名_clean'], {}).get('same_place_avg_rank', {}).get(r['place_code'], r['recent_avg_rank_3']), axis=1)
    race_df['days_since_prev'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('days_since_prev', 30.0))
    race_df['horse_career_runs'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('career_runs', 5))
    race_df['prev_is_minami'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_is_minami', 0))

    trainer_col = '調教師' if '調教師' in race_df.columns else '騎手'
    race_df['騎手_clean'] = race_df['騎手'].astype(str).apply(clean_horse_name)
    race_df['trainer_clean'] = race_df[trainer_col].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip() if pd.notna(x) else '')
    race_df['owner_clean'] = race_df['馬主'].astype(str).apply(clean_horse_name) if '馬主' in race_df.columns else ''
    race_df['combo_key'] = race_df['騎手_clean'] + "_" + race_df['trainer_clean']

    race_df['jockey_win_rate'] = race_df['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
    race_df['trainer_win_rate'] = race_df['trainer_clean'].apply(lambda x: trainer_dict.get(x, 0.05))
    race_df['combo_win_rate'] = race_df['combo_key'].apply(lambda x: combo_dict.get(x, 0.05))
    race_df['owner_win_rate'] = race_df['owner_clean'].apply(lambda x: owner_dict.get(x, 0.05))
    race_df['waku_place_win_rate'] = race_df.apply(lambda r: waku_place_dict.get((r['place_code'], r['distance_num'], r['waku_num']), 0.10), axis=1)
    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)

    raw_time = 75.0 - (race_df['recent_avg_rank_3'].clip(1, 14) - 3.0) * 3.5 + (race_df['斤量'] - 54.0) * 1.5
    raw_start = (12.0 - race_df['first_corner'].clip(upper=10.0)) * 6.5
    race_df['custom_time_index'] = pd.Series(raw_time).fillna(30.0).clip(20.0, 99.0).round(1)
    race_df['custom_start_index'] = pd.Series(raw_start).fillna(30.0).clip(20.0, 99.0).round(1)

    race_df['first_corner_rank'] = race_df['first_corner'].rank(ascending=True, method='min')
    race_df['is_nige_candidate'] = (race_df['prev_1c'] <= 2.5).astype(int)
    nige_count = race_df['is_nige_candidate'].sum()
    race_df['single_escape_flag'] = np.where((race_df['is_nige_candidate'] == 1) & (nige_count == 1), 1, 0)

    race_df['place_prob'] = 0.3
    race_df['win_prob'] = 0.1
    
    if model_data and isinstance(model_data, dict):
        try:
            m_feat = model_data.get('features', [])
            m_place = model_data.get('model_place')
            m_win = model_data.get('model_win')
            
            if m_feat and m_place:
                X = race_df.copy()
                for f in m_feat:
                    if f not in X.columns: X[f] = 0.0
                X_input = X[m_feat]
                
                race_df['place_prob'] = m_place.predict_proba(X_input)[:, 1]
                race_df['win_prob'] = m_win.predict_proba(X_input)[:, 1]
        except Exception: pass

    # ⭕ 修正後の正常なコード
max_p = race_df['place_prob'].max()
if max_p > 0:
    # トップ馬を99点にし、相対的な能力差を10〜99点にきれいに分散させる
    race_df['score_brain1'] = ((race_df['place_prob'] / max_p) * 89 + 10).clip(10, 99).astype(int)
else:
    race_df['score_brain1'] = 50

    race_df['expected_odds'] = (1.0 / race_df['win_prob'].clip(lower=0.01)).round(1)
    race_df['is_abnormal'] = (race_df['単勝_num'] < race_df['expected_odds'] * 0.5) & (race_df['単勝_num'] >= 4.0)

    return race_df.sort_values(by=['score_brain1', 'place_prob'], ascending=[False, False]).reset_index(drop=True)

st.sidebar.header("🔄 画面の更新")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
if st.sidebar.button("🔄 キャッシュ完全クリア＆リロード", use_container_width=True): 
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# 印を最大5頭（◎◯▲△⭐）に設定
def get_mark(idx):
    if idx == 0: return "◎ 軸馬"
    elif idx == 1: return "◯ 相手"
    elif idx == 2: return "▲ 相手"
    elif idx == 3: return "△ 相手"
    elif idx == 4: return "⭐ 相手"
    else: return "消"

def generate_beautiful_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>馬体重</th><th>注目フラグ</th><th>騎手</th><th>脚質</th><th>純粋AI能力</th><th>3着内率</th><th>1着率</th><th>単勝オッズ</th><th>異常</th><th>印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        mark = get_mark(i)
        b_cls = "badge-honmei" if "◎" in mark else "badge-taikou" if "◯" in mark else "badge-tana" if "▲" in mark else "badge-renka" if "△" in mark else "badge-ana" if "⭐" in mark else "badge-keshi"
        kyaku = r.get('脚質', '-')
        k_style = "background:#ff7675;" if kyaku == "逃" else "background:#e67e22;" if kyaku == "先" else "background:#3498db;" if kyaku == "差" else "background:#2ecc71;"

        abnormal = "<span class='badge-alert'>🚨大口</span>" if r.get('is_abnormal', False) else "-"
        
        actual_w = str(r.get('馬体重', '-')).strip()
        if actual_w in ["", "-", "nan", "NaN", "None"]:
            body_w = int(r.get('body_weight', 470))
            weight_str = f"<span style='color:#aaa; font-size:0.9em;'>{body_w}<br>(前走)</span>"
        else:
            weight_str = f"<b>{format_weight_display(actual_w)}</b>"
        
        flags = []
        if r.get('single_escape_flag', 0) == 1: flags.append("<span style='color:#e74c3c; font-size:0.85em; font-weight:bold;'>[🔥] 独走逃げ</span>")
        elif r.get('first_corner_rank', 99) <= 2: flags.append("<span style='color:#e67e22; font-size:0.85em; font-weight:bold;'>[⚡] テン速</span>")
        if r.get('jockey_win_rate', 0) >= 0.15: flags.append("<span style='color:#8e44ad; font-size:0.85em; font-weight:bold;'>[👑] 凄腕</span>")
        if r.get('trainer_win_rate', 0) >= 0.15: flags.append("<span style='color:#2980b9; font-size:0.85em; font-weight:bold;'>[🏢] 名門</span>")
        if r.get('combo_win_rate', 0) >= 0.20: flags.append("<span style='color:#d35400; font-size:0.85em; font-weight:bold;'>[🤝] 黄金コンビ</span>")
        if r.get('corner_diff', 0) >= 2.0: flags.append("<span style='color:#16a085; font-size:0.85em; font-weight:bold;'>[🌀] マクリ脚</span>")
        
        flag_str = "<br>".join(flags) if flags else "-"

        html += f"""<tr>
<td style='font-weight:bold; color:#c94a65 !important;'>{int(r['馬番_num']):02d}</td>
<td style='text-align:left; font-weight:800; color:#5a3d46 !important;'>{r.get('馬名', '-')}</td>
<td>{weight_str}</td>
<td style='line-height:1.3;'>{flag_str}</td>
<td style='color:#666666 !important;'>{r.get('騎手', '-')}</td>
<td><span style='{k_style} color:#fff !important; padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{kyaku}</span></td>
<td style='color:#5a3d46 !important;'><b>{int(r['score_brain1'])}点</b></td>
<td style='color:#c94a65 !important;'><b>{r['place_prob']*100:.1f}%</b></td>
<td style='color:#666666 !important;'>{r['win_prob']*100:.1f}%</td>
<td style='color:#666666 !important;'>{r['単勝_num']:.1f}倍</td>
<td>{abnormal}</td>
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
                        if cols[j].button(f"{r}R", key=f"btn_{rid}", use_container_width=True, type="primary" if st.session_state['selected_race_id'] == rid else "secondary"):
                            st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id'] and not df_future.empty:
        st.markdown("---")
        target_id = str(st.session_state['selected_race_id'])
        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0]
        race_display_name = f"{info['place_name']} {info['r_num']}R 【{info.get('race_name', '')}】"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        cond = st.radio("🌧️ 現在の馬場状態を選択してください", ["良", "稍重", "重", "不良"], horizontal=True, index=["良", "稍重", "重", "不良"].index(st.session_state['baba_status']))
        if cond != st.session_state['baba_status']: 
            st.session_state['baba_status'] = cond
            st.rerun()
        
        scored_df = calculate_race_scores(target_id, df_future, baba_status=st.session_state['baba_status'])
        
        if scored_df is not None:
            # --------------------------------------------------------
            # 🚨【1軸4頭流し専用】正しく整合性を合わせた判定ロジック
            # --------------------------------------------------------
            scores = scored_df['score_brain1'].tolist()
            s1 = scores[0] if len(scores) > 0 else 50
            s2 = scores[1] if len(scores) > 1 else 40
            s5 = scores[4] if len(scores) > 4 else 15
            
            diff1_2 = s1 - s2
            diff1_5 = s1 - s5  # 上位5頭（軸1+相手4）内のスコア差

            # 上位5頭の中でも実力が完全に拮抗していて絞れない場合のみスルー
            if diff1_5 <= 8:
                st.markdown("""
                <div class='rec-banner-through'>
                    🚨 判定：【見（スルー）推奨】（上位拮抗・大混戦）<br>
                    <span style='font-size:0.85em; font-weight:normal;'>上位5頭（軸＋相手4頭）の実力が横一線で拮抗しており危険です。購入せずスルーするのが期待値上、最も安全です。</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                u_jiku = int(scored_df.iloc[0]['馬番_num'])
                u_a1 = int(scored_df.iloc[1]['馬番_num']) if len(scored_df) > 1 else 0
                u_a2 = int(scored_df.iloc[2]['馬番_num']) if len(scored_df) > 2 else 0
                u_a3 = int(scored_df.iloc[3]['馬番_num']) if len(scored_df) > 3 else 0
                u_a4 = int(scored_df.iloc[4]['馬番_num']) if len(scored_df) > 4 else 0
                
                st.markdown(f"""
                <div class='rec-banner-one-strong'>
                    🎯 判定：【高的中・6点勝負】三連複 軸1頭相手4頭流し (計6点 / 600円)<br>
                    <span style='font-size:0.85em; font-weight:normal;'>
                    * <b>軸馬:</b> <b>{u_jiku:02d}</b><br>
                    * <b>相手馬 (4頭):</b> {u_a1:02d}, {u_a2:02d}, {u_a3:02d}, {u_a4:02d}<br>
                    * <b>買い目 (6点):</b> ({u_jiku:02d}-{u_a1:02d}-{u_a2:02d}), ({u_jiku:02d}-{u_a1:02d}-{u_a3:02d}), ({u_jiku:02d}-{u_a1:02d}-{u_a4:02d}), ({u_jiku:02d}-{u_a2:02d}-{u_a3:02d}), ({u_jiku:02d}-{u_a2:02d}-{u_a4:02d}), ({u_jiku:02d}-{u_a3:02d}-{u_a4:02d})
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # --------------------------------------------------------
            # 📊 出馬表（スコア一覧表）
            # --------------------------------------------------------
            st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAIスコア (📍 軸1頭相手4頭厳選)</div>", unsafe_allow_html=True)
            st.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        # --------------------------------------------------------
        # 🎀 Gemini詳細解説（見解を補足したい場合のみクリック）
        # --------------------------------------------------------
        if st.button("🎀 Geminiの見解（解説テキスト）を生成する", use_container_width=True):
            if not api_key_input: 
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            scores = scored_df['score_brain1'].tolist()
            s1 = scores[0] if len(scores) > 0 else 50
            s2 = scores[1] if len(scores) > 1 else 40
            s5 = scores[4] if len(scores) > 4 else 15
            
            diff1_2 = s1 - s2
            diff1_5 = s1 - s5

            if diff1_5 <= 8:
                rec_pattern_name = "🚨 カオス混戦・【見（スルー）推奨】"
            else:
                rec_pattern_name = "🎯 【高的中・6点勝負】三連複 軸1頭相手4頭流し (6点)"

            table_summary = []
            for idx, row in scored_df.iterrows():
                mark = get_mark(idx)
                if mark != "消":
                    table_summary.append(
                        f"印:{mark} | 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | AI能力スコア:{row['score_brain1']} | 3着内率:{row['place_prob']*100:.1f}%"
                    )

            sys_inst = f"""あなたは地方競馬の勝ち子ちゃんです。
Markdownの見出しタグ（###や---など）は使わず、絵文字混じりの綺麗な文章で回答してください。

競馬場: {info['place_name']} / 馬場: {st.session_state['baba_status']}
自動判定された買い目戦略: {rec_pattern_name}

【回答の構成】
🌸 勝ち子ちゃんの馬場・展開の見解
（ここに短評を入れる）

🎯 注目馬解説
◎ 軸馬: 馬番・馬名（理由）
◯ 相手1: 馬番・馬名（理由）
▲ 相手2: 馬番・馬名（理由）
△ 相手3: 馬番・馬名（理由）
⭐ 相手4: 馬番・馬名（理由）
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