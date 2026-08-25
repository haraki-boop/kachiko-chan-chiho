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
# 🌸 1. カスタムCSS & デザイン設定 (馬単専用脳)
# ==========================================
st.set_page_config(page_title="AI予想 勝ち子ちゃん | 馬単フォーメーション 特化脳", page_icon="🌸", layout="wide")

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
    .badge-alert { background: linear-gradient(135deg, #e1b12c, #fbc531); color: #000 !important; font-weight: 900; padding: 3px 8px; border-radius: 6px; }
    .gemini-output-box { background-color: #ffffff !important; color: #222222 !important; padding: 20px; border-radius: 12px; border: 2px solid #f2cdd5; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (馬単フォーメーション 特化脳)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'baba_status' not in st.session_state: st.session_state['baba_status'] = "良"

def set_race_id(rid):
    st.session_state['selected_race_id'] = rid

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HISTORY_CSV, FUTURE_CSV, ML_TARGET_CSV, MODEL_FILE = "prediction_history_chiho.csv", "future_races_chiho.csv", "ml_target_data_chiho.csv", "keiba_ai_model_nar.pkl"
NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

def clean_horse_name(name): 
    return re.sub(r'[\s\u3000]+', '', unicodedata.normalize('NFKC', str(name))) if not pd.isna(name) else ""

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
    jockey_dict, jockey_runs_dict, trainer_dict, combo_dict, owner_dict, horse_dict, waku_place_dict = {}, {}, {}, {}, {}, {}, {}
    if not df_p.empty:
        df_p['馬名_clean'] = df_p['馬名'].astype(str).apply(clean_horse_name)
        df_p['first_corner'] = pd.to_numeric(df_p.get('first_corner', df_p.get('1角')), errors='coerce').fillna(8.0)
        df_p['last_corner'] = pd.to_numeric(df_p.get('last_corner', df_p.get('4角')), errors='coerce').fillna(df_p['first_corner'])
        df_p['last_3f'] = pd.to_numeric(df_p.get('last_3f', df_p.get('上り')), errors='coerce').fillna(39.0)
        df_p['time_sec'] = pd.to_numeric(df_p.get('time_sec', df_p.get('タイム')), errors='coerce')
        df_p['distance_num'] = pd.to_numeric(df_p.get('distance'), errors='coerce').fillna(1400)
        df_p['speed'] = df_p['distance_num'] / df_p['time_sec'].clip(lower=10.0)
        df_p['place_code'] = df_p['race_id'].astype(str).str[4:6]
        df_p['waku_num'] = pd.to_numeric(df_p.get('枠番'), errors='coerce').fillna(0)
        df_p['騎手_clean'] = df_p['騎手'].astype(str).apply(clean_horse_name)
        trainer_col = '調教師' if '調教師' in df_p.columns else '騎手'
        df_p['trainer_clean'] = df_p[trainer_col].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip() if pd.notna(x) else '')
        df_p['prize_money'] = pd.to_numeric(df_p.get('賞金(万円)'), errors='coerce').fillna(0.0)

        if 'target_win' in df_p.columns:
            for j, m in df_p.groupby('騎手_clean')['target_win'].mean().items(): jockey_dict[j] = m
            for j, c in df_p.groupby('騎手_clean')['target_win'].count().items(): jockey_runs_dict[j] = c
            for wp, m in df_p.groupby(['place_code', 'distance_num', 'waku_num'])['target_win'].mean().items(): waku_place_dict[wp] = m

        df_p['date_dt'] = pd.to_datetime(df_p.get('date', pd.Series(['2020-01-01']*len(df_p))), errors='coerce')
        for h, group in df_p.sort_values('date_dt').groupby('馬名_clean'):
            r3 = group.tail(3)
            f_c = pd.to_numeric(r3.get('first_corner', pd.Series([8.0])), errors='coerce').fillna(8.0).mean()
            l_c = pd.to_numeric(r3.get('last_corner', pd.Series([8.0])), errors='coerce').fillna(f_c).mean()
            
            horse_dict[h] = {
                'first_corner': f_c,
                'corner_diff': f_c - l_c,
                'best_speed': group['speed'].max() if len(group) > 0 else 15.0,
                'recent_avg_rank_3': r3.get('target_rank', pd.Series([6.0])).mean()
            }
    return jockey_dict, jockey_runs_dict, trainer_dict, combo_dict, owner_dict, horse_dict, waku_place_dict

jockey_dict, jockey_runs_dict, trainer_dict, combo_dict, owner_dict, horse_dict, waku_place_dict = build_past_dicts(df_past)

def get_kyakushitsu(fc): 
    return "逃" if fc <= 2.0 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

def calculate_race_scores(race_id_target, target_df, baba_status="良"):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy().reset_index(drop=True)
    if race_df.empty: return None

    race_df['place_code'] = pd.to_numeric(race_df['race_id'].astype(str).str[4:6], errors='coerce').fillna(0.0)
    race_df['単勝_num'] = pd.to_numeric(race_df['単勝'], errors='coerce').fillna(15.0) if '単勝' in race_df.columns else 15.0
    race_df['distance_num'] = pd.to_numeric(race_df['distance'], errors='coerce').fillna(1400) if 'distance' in race_df.columns else 1400
    race_df['weight_num'] = pd.to_numeric(race_df['斤量'], errors='coerce').fillna(54.0) if '斤量' in race_df.columns else 54.0
    race_df['馬番_num'] = pd.to_numeric(race_df['馬番'], errors='coerce').fillna(0) if '馬番' in race_df.columns else 0
    race_df['waku_num'] = pd.to_numeric(race_df['枠番'], errors='coerce').fillna(0) if '枠番' in race_df.columns else 0
    race_df['馬名_clean'] = race_df['馬名'].astype(str).apply(clean_horse_name)

    if '馬体重' in race_df.columns:
        parsed_w = race_df['馬体重'].apply(parse_weight_info)
        race_df['horse_weight'] = [x[0] for x in parsed_w]
    else: 
        race_df['horse_weight'] = 470.0

    race_df['first_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('first_corner', 6.0))
    race_df['corner_diff'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('corner_diff', 0.0))
    race_df['is_nige_candidate'] = (race_df['first_corner'] <= 2.5).astype(int)
    nige_count = race_df['is_nige_candidate'].sum()
    race_df['single_escape_flag'] = np.where((race_df['is_nige_candidate'] == 1) & (nige_count == 1), 1, 0)

    race_df['騎手_clean'] = race_df['騎手'].astype(str).apply(clean_horse_name)
    race_df['jockey_win_rate'] = race_df['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)
    race_df['first_corner_rank'] = race_df['first_corner'].rank(ascending=True, method='min')

    race_df['place_prob'] = 0.3
    race_df['win_prob'] = 0.1
    
    if model_data and isinstance(model_data, dict):
        try:
            m_feat = model_data.get('features', [])
            m_place = model_data.get('model_place')
            m_win = model_data.get('model_win')
            
            if m_feat and m_place and m_win:
                X = race_df.copy()
                for f in m_feat:
                    if f not in X.columns: X[f] = 0.0
                X_input = X[m_feat]
                
                race_df['place_prob'] = m_place.predict_proba(X_input)[:, 1]
                race_df['win_prob'] = m_win.predict_proba(X_input)[:, 1]
        except Exception: pass

    # ==========================================
    # 🧠 馬単特化型ブレインチューニング (スコア再計算)
    # ==========================================
    w_max, w_min = race_df['win_prob'].max(), race_df['win_prob'].min()
    p_max, p_min = race_df['place_prob'].max(), race_df['place_prob'].min()
    
    w_norm = (race_df['win_prob'] - w_min) / (w_max - w_min + 1e-6)
    p_norm = (race_df['place_prob'] - p_min) / (p_max - p_min + 1e-6)
    
    # 馬単専用ボーナス：テンの速い馬（1角3番手以内）にアドバンテージ
    front_bonus = np.where(race_df['first_corner'] <= 3.0, 0.15, 0.0)
    
    # 1着至上主義ハイブリッドスコア: 勝率(60%) + 3着内率(30%) + 展開利(15%)
    raw_score = (w_norm * 0.60) + (p_norm * 0.30) + front_bonus
    
    r_max, r_min = raw_score.max(), raw_score.min()
    if r_max > r_min:
        race_df['score_brain1'] = (((raw_score - r_min) / (r_max - r_min)) * 89 + 10).astype(int)
    else:
        race_df['score_brain1'] = 50

    race_df['expected_odds'] = (1.0 / race_df['win_prob'].clip(lower=0.01)).round(1)
    race_df['is_abnormal'] = (race_df['単勝_num'] < race_df['expected_odds'] * 0.5) & (race_df['単勝_num'] >= 4.0)

    # ソート順を「馬単用スコア」と「1着率」を最優先に変更
    return race_df.sort_values(by=['score_brain1', 'win_prob'], ascending=[False, False]).reset_index(drop=True)

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
    else: return "消"

def generate_beautiful_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>馬体重</th><th>注目フラグ</th><th>騎手</th><th>脚質</th><th>馬単用AI能力</th><th>1着率</th><th>3着内率</th><th>単勝オッズ</th><th>異常</th><th>印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        mark = get_mark(i)
        b_cls = "badge-honmei" if "◎" in mark else "badge-taikou" if "◯" in mark else "badge-tana" if "▲" in mark else "badge-renka" if "△" in mark else "badge-keshi"
        kyaku = r.get('脚質', '-')
        k_style = "background:#ff7675;" if kyaku == "逃" else "background:#e67e22;" if kyaku == "先" else "background:#3498db;" if kyaku == "差" else "background:#2ecc71;"

        abnormal = "<span class='badge-alert'>🚨大口</span>" if r.get('is_abnormal', False) else "-"
        
        actual_w = str(r.get('馬体重', '-')).strip()
        if actual_w in ["", "-", "nan", "NaN", "None"]:
            body_w = int(r.get('horse_weight', 470))
            weight_str = f"<span style='color:#aaa; font-size:0.9em;'>{body_w}<br>(前走)</span>"
        else:
            weight_str = f"<b>{format_weight_display(actual_w)}</b>"
        
        flags = []
        if r.get('single_escape_flag', 0) == 1: flags.append("<span style='color:#e74c3c; font-size:0.85em; font-weight:bold;'>[🔥] 独走逃げ</span>")
        elif r.get('first_corner_rank', 99) <= 2: flags.append("<span style='color:#e67e22; font-size:0.85em; font-weight:bold;'>[⚡] テン速</span>")
        if r.get('jockey_win_rate', 0) >= 0.15: flags.append("<span style='color:#8e44ad; font-size:0.85em; font-weight:bold;'>[👑] 凄腕</span>")
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
<td style='color:#c94a65 !important; font-size:1.05em;'><b>{r['win_prob']*100:.1f}%</b></td>
<td style='color:#666666 !important;'>{r['place_prob']*100:.1f}%</td>
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
                        cols[j].button(
                            f"{r}R", 
                            key=f"btn_{rid}", 
                            use_container_width=True, 
                            type="primary" if st.session_state['selected_race_id'] == rid else "secondary",
                            on_click=set_race_id, args=(rid,)
                        )

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
            def safe_idx(df, idx):
                return int(df.iloc[idx]['馬番_num']) if len(df) > idx else int(df.iloc[-1]['馬番_num'])

            u_1 = safe_idx(scored_df, 0)
            u_2 = safe_idx(scored_df, 1)
            u_3 = safe_idx(scored_df, 2)
            u_4 = safe_idx(scored_df, 3)

            st.markdown(f"""
            <div class='rec-banner-formation'>
                🎯 判定：【黄金・馬単フォーメーション】 1・2位 ➔ 1〜4位 (計6点 / 600円)<br>
                <span style='font-size:0.85em; font-weight:normal;'>
                * <b>1着候補 (2頭):</b> <b>{u_1:02d}, {u_2:02d}</b><br>
                * <b>2着候補 (4頭):</b> {u_1:02d}, {u_2:02d}, {u_3:02d}, {u_4:02d}<br>
                * <b>買い目 (6点):</b> ({u_1:02d}➔{u_2:02d}), ({u_1:02d}➔{u_3:02d}), ({u_1:02d}➔{u_4:02d}), ({u_2:02d}➔{u_1:02d}), ({u_2:02d}➔{u_3:02d}), ({u_2:02d}➔{u_4:02d})<br>
                * <b>理由:</b> 馬単特化脳により「勝つ力」と「テンの速さ」を持つ上位4頭を抽出。取りこぼしなく仕留めます。
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんの馬単用AIスコア (📍 1着・2着 狙い撃ち)</div>", unsafe_allow_html=True)
            st.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Geminiの見解（解説テキスト）を生成する", use_container_width=True):
            if not api_key_input: 
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            rec_pattern_name = "🎯 【黄金・馬単フォーメーション】 1・2位 ➔ 1〜4位 (6点)"

            table_summary = []
            for idx, row in scored_df.iterrows():
                mark = get_mark(idx)
                if mark != "消":
                    table_summary.append(
                        f"印:{mark} | 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | AI能力スコア:{row['score_brain1']} | 1着率:{row['win_prob']*100:.1f}%"
                    )

            sys_inst = f"""あなたは地方競馬の勝ち子ちゃんです。
Markdownの見出しタグ（###や---など）は使わず、絵文字混じりの綺麗な文章で回答してください。

競馬場: {info['place_name']} / 馬場: {st.session_state['baba_status']}
自動判定された買い目戦略（馬単フォーメーション）: {rec_pattern_name}

【回答の構成】
🌸 勝ち子ちゃんの馬場・展開の見解
（ここに短評を入れる）

🎯 注目馬解説 (馬単1着・2着の観点で)
◎ 本命: 馬番・馬名（理由）
◯ 対抗: 馬番・馬名（理由）
▲ 単穴: 馬番・馬名（理由）
△ 連下: 馬番・馬名（理由）
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