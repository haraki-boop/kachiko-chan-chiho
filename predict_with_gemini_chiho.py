import os
import re
import time
import requests
import pandas as pd
import numpy as np
import joblib
import unicodedata
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
from google import genai
from google.genai import types

# ==========================================
# 🌸 1. 勝ち子ちゃん専用 カスタムCSS & デザイン設定
# ==========================================
st.set_page_config(page_title="AI予想 勝ち子ちゃん | 地方完全版", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fcf9f9 !important; color: #333333 !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    p, span, label, div, li, td, th { color: #333333; }
    h1 { font-size: 1.9rem !important; color: #c94a65 !important; font-weight: 800; }
    h2 { font-size: 1.4rem !important; color: #5a3d46 !important; }
    h3 { font-size: 1.2rem !important; color: #c94a65 !important; }
    .section-header { font-size: 1.25rem; font-weight: 800; color: #c94a65 !important; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #f2cdd5; padding-bottom: 6px; }
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
    .badge-alert { background: linear-gradient(135deg, #e1b12c, #fbc531); color: #000 !important; font-weight: 900; padding: 3px 8px; border-radius: 6px; animation: blink 1.5s infinite; }
    .gemini-output-box { background-color: #ffffff !important; color: #222222 !important; padding: 20px; border-radius: 12px; border: 2px solid #f2cdd5; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (純粋勝率＆期待値特化)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'baba_status' not in st.session_state: st.session_state['baba_status'] = "良"

# ==========================================
# 🔑 2. 定数と環境設定・読み込み
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HISTORY_CSV, FUTURE_CSV, ML_TARGET_CSV, MODEL_FILE = "prediction_history_chiho.csv", "future_races_chiho.csv", "ml_target_data_chiho.csv", "keiba_ai_model_nar.pkl"
NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}
TRACK_BIAS = {"浦和": {"逃": 1.25, "先": 1.15, "差": 0.80, "追": 0.70}, "大井": {"逃": 0.90, "先": 1.00, "差": 1.20, "追": 1.15}, "川崎": {"逃": 1.15, "先": 1.10, "差": 0.95, "追": 0.85}, "船橋": {"逃": 1.05, "先": 1.05, "差": 1.05, "追": 0.95}, "園田": {"逃": 1.20, "先": 1.15, "差": 0.85, "追": 0.75}}

def clean_horse_name(name): return re.sub(r'[\s\u3000]+', '', unicodedata.normalize('NFKC', str(name))) if not pd.isna(name) else ""

def load_csv_utf8(path, dtype_dict=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try: return pd.read_csv(path, dtype=dtype_dict, encoding=enc)
        except: continue
    return pd.DataFrame()

@st.cache_resource
def load_model(): return joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None

df_past = load_csv_utf8(ML_TARGET_CSV, {'race_id': str})
df_future = load_csv_utf8(FUTURE_CSV, {'race_id': str, '馬番': str})
df_history = load_csv_utf8(HISTORY_CSV, {'race_id': str})

if not df_past.empty:
    def _parse_rank(x):
        if pd.isna(x): return 6.0
        try: return float(str(x).replace('着', '').replace('(', '').replace(')', '').strip())
        except: return 6.0
        
    if '着順' in df_past.columns:
        df_past['target_rank'] = df_past['着順'].apply(_parse_rank)
    elif 'target_rank' in df_past.columns:
        df_past['target_rank'] = df_past['target_rank'].apply(_parse_rank)
        
    if 'target_win' not in df_past.columns and 'target_rank' in df_past.columns:
        df_past['target_win'] = (df_past['target_rank'] == 1.0).astype(int)

if not df_future.empty and 'race_id' in df_future.columns:
    df_future['place_code'] = df_future['race_id'].str[4:6]
    df_future['place_name'] = df_future['place_code'].map(NAR_PLACES).fillna("地方")
    df_future['r_num'] = df_future['race_id'].str[10:12].astype(int)
    df_future['day_label'] = df_future['date'] if 'date' in df_future.columns else datetime.now().strftime("%Y-%m-%d")

model_data = load_model()

@st.cache_data
def build_past_dicts(df_p):
    jockey_dict, horse_dict = {}, {}
    if not df_p.empty:
        if 'target_win' in df_p.columns:
            for j, row in df_p.groupby('騎手')['target_win'].agg(['count', 'mean']).iterrows(): jockey_dict[j] = {'j_runs': row['count'], 'j_win_rate': row['mean']}
        df_p['馬名_clean'] = df_p['馬名'].astype(str).apply(clean_horse_name)
        for h, group in df_p.sort_values('date' if 'date' in df_p.columns else 'race_id').groupby('馬名_clean'):
            r3 = group.tail(3)
            horse_dict[h] = {
                'prev_dist': r3.iloc[-1].get('distance_num', 1400),
                'prev_weight': pd.to_numeric(r3.iloc[-1].get('斤量'), errors='coerce') if '斤量' in r3.columns else 54.0,
                'prev_1c': pd.to_numeric(r3.get('first_corner', pd.Series([8.0])), errors='coerce').fillna(8.0).mean(),
                'recent_avg_rank': r3.get('target_rank', pd.Series([6.0])).mean()
            }
    return jockey_dict, horse_dict

jockey_dict, horse_dict = build_past_dicts(df_past)
def get_kyakushitsu(fc): return "逃" if fc <= 2.0 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

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
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return 470.0, 0.0

# ==========================================
# 3. AIスコア ＆ ピュア勝率算出（数十項目・優先評価搭載）
# ==========================================
def calculate_race_scores(race_id_target, target_df, baba_status="良"):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy().reset_index(drop=True)
    if race_df.empty: return None

    place = race_df['place_name'].iloc[0] if 'place_name' in race_df.columns else "地方"
    bias_dict = TRACK_BIAS.get(place, {"逃": 1.0, "先": 1.0, "差": 1.0, "追": 1.0})

    race_df['単勝_num'] = pd.to_numeric(race_df.get('単勝'), errors='coerce').fillna(15.0)
    race_df['distance_num'] = pd.to_numeric(race_df.get('distance'), errors='coerce').fillna(1400)
    race_df['斤量'] = pd.to_numeric(race_df.get('斤量'), errors='coerce').fillna(54.0)
    race_df.loc[(race_df['斤量'] == race_df['単勝_num']) | (race_df['斤量'] < 48.0) | (race_df['斤量'] > 63.0), '斤量'] = 54.0
    race_df['馬番_num'] = pd.to_numeric(race_df.get('馬番'), errors='coerce').fillna(0)
    race_df['馬名_clean'] = race_df['馬名'].astype(str).apply(clean_horse_name)

    if '性齢' in race_df.columns:
        sex_age_parsed = race_df['性齢'].apply(parse_sex_age)
        race_df['sex_code'] = [x[0] for x in sex_age_parsed]
        race_df['age'] = [x[1] for x in sex_age_parsed]
    else:
        race_df['sex_code'], race_df['age'] = 0, 4.0

    if '馬体重' in race_df.columns:
        weight_parsed = race_df['馬体重'].apply(parse_weight_info)
        race_df['body_weight'] = [x[0] for x in weight_parsed]
        race_df['body_weight_diff'] = [x[1] for x in weight_parsed]
    else:
        race_df['body_weight'], race_df['body_weight_diff'] = 470.0, 0.0

    race_df['kinryo_weight_ratio'] = race_df['斤量'] / race_df['body_weight'].clip(lower=350.0)

    race_df['prev_dist'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_dist', 1400))
    race_df['prev_weight'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_weight', 54.0))
    race_df['first_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_1c', 6.0))
    race_df['recent_avg_rank'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank', 6.0))
    race_df['jockey_win_rate'] = race_df['騎手'].apply(lambda x: jockey_dict.get(x, {}).get('j_win_rate', 0.05))
    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)
    race_df['bias_mult'] = race_df['脚質'].apply(lambda k: bias_dict.get(k, 1.0))

    raw_time = (75.0 - (race_df['recent_avg_rank'].clip(1, 14) - 3.0) * 3.5 + (race_df['斤量'] - 54.0) * 1.5) * (1.0 + (1.0 - race_df['bias_mult'])*0.5)
    raw_start = ((12.0 - race_df['first_corner'].clip(upper=10.0)) * 6.5) * race_df['bias_mult']

    if baba_status in ["重", "不良"]:
        raw_start = np.where(race_df['脚質'].isin(["逃", "先"]), raw_start + 8.0, raw_start - 3.0)
        raw_time = np.where(race_df['脚質'].isin(["逃", "先"]), raw_time + 5.0, raw_time - 2.0)
    elif baba_status == "良":
        raw_time = np.where(race_df['脚質'].isin(["差", "追"]), raw_time + 4.0, raw_time - 2.0)

    race_df['custom_time_index'] = pd.Series(raw_time).fillna(30.0).clip(20.0, 99.0).round(1)
    race_df['custom_start_index'] = pd.Series(raw_start).fillna(30.0).clip(20.0, 99.0).round(1)

    features_to_zscore = ['斤量', 'first_corner', 'jockey_win_rate', 'custom_time_index', 'custom_start_index']
    for col in features_to_zscore:
        mean_val, std_val = race_df[col].mean(), race_df[col].std()
        if pd.isna(std_val) or std_val == 0: std_val = 1e-5
        race_df[f'{col}_zscore'] = (race_df[col] - mean_val) / std_val
        race_df[f'{col}_rank'] = race_df[col].rank(ascending=False, method='min')

    ai_prob = np.zeros(len(race_df))
    if model_data:
        try:
            m_win = model_data.get('model_win', model_data.get('model'))
            X = race_df[model_data['features']].fillna(0.0)
            if hasattr(m_win, "predict_proba"):
                preds = m_win.predict_proba(X)[:, 1]
            else:
                preds = m_win.predict(X)
            
            p_min, p_max = preds.min(), preds.max()
            if p_max > p_min:
                ai_prob = (preds - p_min) / (p_max - p_min)
            else:
                ai_prob = np.zeros(len(race_df))
        except: pass

    time_rank_norm = (race_df['custom_time_index'] - race_df['custom_time_index'].min()) / (race_df['custom_time_index'].max() - race_df['custom_time_index'].min() + 1e-5)
    start_rank_norm = (race_df['custom_start_index'] - race_df['custom_start_index'].min()) / (race_df['custom_start_index'].max() - race_df['custom_start_index'].min() + 1e-5)

    priority_bonus = np.zeros(len(race_df))
    priority_bonus += np.where(race_df['first_corner_rank'] <= 2, 0.25, 0.0)
    priority_bonus += np.where(race_df['jockey_win_rate'] >= 0.15, 0.20, 0.0)
    priority_bonus += np.where(race_df['distance_num'] < race_df['prev_dist'], 0.10, 0.0)
    priority_bonus += np.where(race_df['斤量'] < race_df['prev_weight'], 0.05, 0.0)

    pure_win_score = (ai_prob * 0.3) + (time_rank_norm * 0.3) + (start_rank_norm * 0.2) + priority_bonus
    win_sum = pure_win_score.sum()
    race_df['win_prob'] = pure_win_score / win_sum if win_sum > 0 else 1.0 / len(race_df)

    race_df['ev_brain2'] = (race_df['win_prob'] * race_df['単勝_num']).fillna(0).round(2)
    race_df['expected_odds'] = (1.0 / race_df['win_prob'].clip(lower=0.01)).round(1)
    race_df['is_abnormal'] = (race_df['単勝_num'] < race_df['expected_odds'] * 0.5) & (race_df['単勝_num'] >= 4.0)

    max_p = max(race_df['win_prob'].max(), 0.01)
    ev_score = (race_df['ev_brain2'].clip(0, 3.0) / 3.0) * 35.0 
    prob_score = (race_df['win_prob'] / max_p) * 60.0
    
    raw_score = (prob_score + ev_score).fillna(10).astype(int)
    race_df['score_brain1'] = np.where(race_df['is_abnormal'], raw_score + 15, raw_score)
    race_df['score_brain1'] = race_df['score_brain1'].clip(10, 99)

    return race_df.sort_values(by=['score_brain1', 'recent_avg_rank'], ascending=[False, True]).reset_index(drop=True)

# ==========================================
# 4. UI ＆ テーブル
# ==========================================
st.sidebar.header("🔄 画面の更新")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
if st.sidebar.button("🔄 キャッシュ完全クリア＆リロード", use_container_width=True): st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

def get_mark(idx):
    if idx == 0: return "◎ 本命"
    elif idx == 1: return "◯ 対抗"
    elif idx == 2: return "▲ 単穴"
    elif idx == 3: return "△1 連下"
    elif idx == 4: return "△2 押さえ"
    else: return "消"

def generate_beautiful_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>馬体重</th><th>騎手</th><th>脚質</th><th>優先フラグ</th><th>AIスコア</th><th>勝率</th><th>オッズ</th><th>適正</th><th>期待値</th><th>異常</th><th>印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        mark = get_mark(i)
        b_cls = "badge-honmei" if "◎" in mark else "badge-taikou" if "◯" in mark else "badge-tana" if "▲" in mark else "badge-renka" if "△" in mark else "badge-keshi"
        kyaku = r.get('脚質', '-')
        k_style = "background:#ff7675;" if kyaku == "逃" else "background:#e67e22;" if kyaku == "先" else "background:#3498db;" if kyaku == "差" else "background:#2ecc71;"
        
        flags = []
        if r.get('first_corner_rank', 99) <= 2: flags.append("<span style='color:red; font-size:0.8em;'>[強] テン速</span>")
        if r.get('jockey_win_rate', 0) >= 0.15: flags.append("<span style='color:red; font-size:0.8em;'>[強] 凄腕</span>")
        if r.get('distance_num', 0) < r.get('prev_dist', 0): flags.append("<span style='color:blue; font-size:0.8em;'>[中] 距短</span>")
        flag_str = "<br>".join(flags) if flags else "-"

        abnormal = "<span class='badge-alert'>🚨大口</span>" if r.get('is_abnormal', False) else "-"
        ev_style = "color:#e74c3c !important; font-weight:900;" if r['ev_brain2'] >= 1.0 else "color:#5a3d46;"
        
        weight_str = r.get('馬体重', '-')
        
        html += f"""<tr>
<td style='font-weight:bold; color:#c94a65 !important;'>{int(r['馬番_num']):02d}</td>
<td style='text-align:left; font-weight:800; color:#5a3d46 !important;'>{r.get('馬名', '-')}</td>
<td style='color:#666666 !important;'>{weight_str}</td>
<td style='color:#666666 !important;'>{r.get('騎手', '-')}</td>
<td><span style='{k_style} color:#fff !important; padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{kyaku}</span></td>
<td style='line-height:1.2;'>{flag_str}</td>
<td style='color:#5a3d46 !important;'><b>{int(r['score_brain1'])}点</b></td>
<td style='color:#c94a65 !important;'><b>{r['win_prob']*100:.1f}%</b></td>
<td style='color:#666666 !important;'>{r['単勝_num']:.1f}倍</td>
<td style='color:#bdc3c7 !important; font-size:0.9em;'>{r.get('expected_odds', 0)}倍</td>
<td style='{ev_style}'><b>{r['ev_brain2']:.2f}</b></td>
<td>{abnormal}</td>
<td><span class='badge-mark {b_cls}'>{mark}</span></td>
</tr>"""
    html += "</tbody></table></div>"
    return html

# ==========================================
# 5. メイン UI
# ==========================================
tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想 (完全版)", "📈 地方実戦成績"])

with tab_forecast:
    if df_future.empty: st.warning("⚠️ 本日の出馬表データが存在しません。")
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
        if cond != st.session_state['baba_status']: st.session_state['baba_status'] = cond; st.rerun()
        
        scored_df = calculate_race_scores(target_id, df_future, baba_status=st.session_state['baba_status'])
        
        if scored_df is not None:
            st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAIスコア (📍 {info['place_name']}優先フラグ反映済)</div>", unsafe_allow_html=True)
            st.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Geminiで【3連複6点＆3連単8点フォーメーション】を予想", type="primary", use_container_width=True):
            if not api_key_input: st.error("【設定エラー】APIキーが見つかりません。"); st.stop()

            table_summary = []
            for idx, row in scored_df.iterrows():
                mark = get_mark(idx)
                if mark != "消":
                    table_summary.append(
                        f"印:{mark} | 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | 馬体重:{row.get('馬体重','')} | 脚質:{row['脚質']} | "
                        f"優先フラグ:{row.get('first_corner_rank',99)<=2} (テン速) / {row.get('jockey_win_rate',0)>=0.15} (凄腕騎手) | "
                        f"オッズ:{row['単勝_num']}倍 (適正:{row.get('expected_odds',0)}倍) | 期待値:{row['ev_brain2']:.2f} | 異常投票:{'あり' if row.get('is_abnormal') else 'なし'}"
                    )

            sys_inst = f"""あなたは地方競馬の勝ち子ちゃんです。
競馬場は「{info['place_name']}」、馬場状態は「{st.session_state['baba_status']}」。
バックテストで最も高い投資効果が証明された【Cプラン（2本柱）】の買い目を案内してください。

【出力フォーマット】
---
### 🌸 馬場傾向・優先フラグ考察
* （短く簡潔に展開や考察を記述）

### 🎯 勝ち子ちゃんの印
* ◎ 本命: 〇〇番
* ◯ 対抗: 〇〇番
* ▲ 単穴: 〇〇番
* △1 連下: 〇〇番
* △2 押さえ: 〇〇番

### 🎀 推奨買い目 (Cプラン)
* **【手堅く勝負】3連複 1頭軸流し (6点)**
  * ◎ ＝ ◯, ▲, △1, △2
* **【高配当狙い】3連単 最強フォーメーション (8点)**
  * 1着: ◎, ◯
  * 2着: ◎, ◯, ▲
  * 3着: ◯, ▲, △1, △2
---"""
            with st.spinner("🎀 馬体重・優先フラグから最強フォーメーション生成中..."):
                try:
                    res_text = genai.Client(api_key=api_key_input).models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"対象レース: {sel_date} {race_display_name}\n\n対象馬:\n" + "\n".join(table_summary),
                        config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.3)
                    ).text
                    st.markdown(f"<div class='gemini-output-box'>{res_text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"エラー: {e}")

with tab_dashboard:
    st.markdown("<div class='section-header'>📈 地方実戦成績ダッシュボード</div>", unsafe_allow_html=True)
    if df_history.empty: st.info("まだ履歴がありません。")
    else: st.info("ダッシュボードは稼働中です。(履歴CSVにデータが保存されると成績が表示されます)")