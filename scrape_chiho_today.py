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
st.set_page_config(page_title="AI予想 勝ち子ちゃん | 地方アルティメット版", page_icon="🌸", layout="wide")

st.markdown('''
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
    .kachi-table th { padding: 10px 12px; text-align: center; border-right: 1px solid rgba(255,255,255,0.2); color: #ffffff !important; }
    .kachi-table td { padding: 8px 12px; text-align: center; border-bottom: 1px solid #f2eced; color: #5a3d46 !important; font-weight: 500; }
    .kachi-table tbody tr:hover td { background: #fff5f7; }
    .badge-mark { color: #fff !important; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 0.85em; display: inline-block; min-width: 55px; }
    .badge-honmei { background: linear-gradient(135deg, #ff4757, #ff6b81); }
    .badge-taikou { background: linear-gradient(135deg, #3742fa, #5352ed); }
    .badge-tana   { background: linear-gradient(135deg, #2ed573, #7bed9f); }
    .badge-renka  { background: linear-gradient(135deg, #ffa502, #eccc68); color: #222 !important; }
    .badge-keshi  { background: #e0e0e0; color: #666666 !important; }
    .badge-idx { background-color: #fff0f3; color: #c94a65 !important; font-weight: 800; padding: 3px 8px; border-radius: 6px; font-size: 0.85em; border: 1px solid #f2cdd5; }
    .badge-alert { background: linear-gradient(135deg, #e1b12c, #fbc531); color: #000 !important; font-weight: 900; padding: 3px 8px; border-radius: 6px; animation: blink 1.5s infinite; }
    @keyframes blink { 0% {opacity: 1;} 50% {opacity: 0.7;} 100% {opacity: 1;} }
    .gemini-output-box { background-color: #ffffff !important; color: #222222 !important; padding: 20px; border-radius: 12px; border: 2px solid #f2cdd5; margin-top: 15px; }
    .gemini-output-box * { color: #222222 !important; }
</style>
''', unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (地方アルティメット版)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'baba_status' not in st.session_state: st.session_state['baba_status'] = "良"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HISTORY_CSV, FUTURE_CSV, ML_TARGET_CSV, MODEL_FILE = "prediction_history_chiho.csv", "future_races_chiho.csv", "ml_target_data_chiho.csv", "keiba_ai_model_nar.pkl"

NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

# 📍 1. 競馬場ごとのコースバイアス定義（逃・先・差・追の有利不利乗数）
TRACK_BIAS = {
    "浦和": {"逃": 1.25, "先": 1.15, "差": 0.80, "追": 0.70}, # 超・前有利
    "園田": {"逃": 1.20, "先": 1.15, "差": 0.85, "追": 0.75}, # 前有利
    "姫路": {"逃": 1.20, "先": 1.15, "差": 0.85, "追": 0.75},
    "大井": {"逃": 0.90, "先": 1.00, "差": 1.20, "追": 1.15}, # 直線長く差し有利
    "門別": {"逃": 0.95, "先": 1.05, "差": 1.15, "追": 1.10}, # 差しが決まる
    "川崎": {"逃": 1.15, "先": 1.10, "差": 0.95, "追": 0.85},
    "船橋": {"逃": 1.05, "先": 1.05, "差": 1.05, "追": 0.95}, # フラット
    "佐賀": {"逃": 1.20, "先": 1.15, "差": 0.85, "追": 0.70}, # イン前絶対有利
    "高知": {"逃": 1.10, "先": 1.10, "差": 1.00, "追": 0.90}, # 内ラチ沿い深く一発あり
}

def clean_horse_name(name): return re.sub(r'[\s\u3000]+', '', unicodedata.normalize('NFKC', str(name))) if not pd.isna(name) else ""

def load_csv_utf8(path, dtype_dict=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'euc-jp']:
        try: return pd.read_csv(path, dtype=dtype_dict, encoding=enc)
        except: continue
    return pd.DataFrame()

@st.cache_resource
def load_model(): return joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else None

df_past, df_future, df_history = load_csv_utf8(ML_TARGET_CSV, {'race_id': str}), load_csv_utf8(FUTURE_CSV, {'race_id': str, '馬番': str}), load_csv_utf8(HISTORY_CSV, {'race_id': str})
if not df_future.empty:
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
        past = df_p.sort_values(['馬名_clean', 'date']) if 'date' in df_p.columns else df_p
        for h, group in past.groupby('馬名_clean'):
            r3 = group.tail(3)
            horse_dict[h] = {'prev_dist': r3.iloc[-1].get('distance_num', 1400), 'prev_1c': r3.get('first_corner', pd.Series([8.0])).mean(), 'recent_avg_rank': r3.get('target_rank', pd.Series([6.0])).mean()}
    return jockey_dict, horse_dict

jockey_dict, horse_dict = build_past_dicts(df_past)

def get_kyakushitsu(fc): return "逃" if fc <= 2.0 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

def calculate_race_scores(race_id_target, target_df, baba_status="良"):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy().reset_index(drop=True)
    if race_df.empty: return None

    place = race_df['place_name'].iloc[0] if 'place_name' in race_df.columns else "地方"
    bias_dict = TRACK_BIAS.get(place, {"逃": 1.0, "先": 1.0, "差": 1.0, "追": 1.0})

    race_df['単勝_num'] = pd.to_numeric(race_df.get('単勝'), errors='coerce').fillna(15.0)
    race_df['weight_num'] = pd.to_numeric(race_df.get('斤量'), errors='coerce').fillna(54.0)
    race_df.loc[(race_df['weight_num'] == race_df['単勝_num']) | (race_df['weight_num'] < 48) | (race_df['weight_num'] > 63), 'weight_num'] = 54.0
    race_df['馬番_num'] = pd.to_numeric(race_df.get('馬番'), errors='coerce').fillna(0)
    race_df['馬名_clean'] = race_df['馬名'].astype(str).apply(clean_horse_name)
    race_df['first_corner'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('prev_1c', 6.0))
    race_df['recent_avg_rank'] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get('recent_avg_rank', 6.0))
    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)
    
    # 📍 コースバイアス適用
    race_df['bias_mult'] = race_df['脚質'].apply(lambda k: bias_dict.get(k, 1.0))

    raw_start = ((12.0 - race_df['first_corner'].clip(upper=10.0)) * 6.5 + (15.0 - race_df['単勝_num'].clip(upper=20.0)) * 0.3) * race_df['bias_mult']
    raw_time = (75.0 - (race_df['recent_avg_rank'].clip(1, 14) - 3.0) * 3.0 + (race_df['weight_num'] - 54.0) * 1.5) * (1.0 + (1.0 - race_df['bias_mult'])*0.5)

    if baba_status in ["重", "不良"]:
        raw_start = np.where(race_df['脚質'].isin(["逃", "先"]), raw_start + 6.0, raw_start - 2.0)
        raw_time = np.where(race_df['脚質'].isin(["逃", "先"]), raw_time + 4.0, raw_time - 1.0)
    elif baba_status == "良":
        raw_time = np.where(race_df['脚質'].isin(["差", "追"]), raw_time + 3.0, raw_time - 1.0)

    race_df['custom_time_index'] = pd.Series(raw_time).fillna(30.0).clip(30.0, 99.0).round(1)
    race_df['custom_start_index'] = pd.Series(raw_start).fillna(30.0).clip(30.0, 99.0).round(1)

    inv_odds = 1.0 / race_df['単勝_num'].clip(lower=1.0)
    base_prob = np.power(inv_odds, 1.2)
    
    if model_data and 'model' in model_data:
        try:
            X = race_df[model_data['features']].fillna(0.0)
            preds = model_data['model'].predict(X)
            base_prob = base_prob * 0.5 + np.power((preds - preds.min()) / (preds.max() - preds.min() + 1e-8) + 0.1, 1.5) * 0.5
        except: pass

    race_df['win_prob'] = (base_prob + (race_df['単勝_num'] >= 10.0).astype(int) * 0.05).fillna(0)
    win_sum = race_df['win_prob'].sum()
    race_df['win_prob'] = race_df['win_prob'] / win_sum if win_sum > 0 else 1.0 / len(race_df)

    # 💸 異常オッズ（大口投票）検知
    race_df['expected_odds'] = (1.0 / race_df['win_prob'].clip(lower=0.01)).round(1)
    race_df['is_abnormal'] = (race_df['単勝_num'] < race_df['expected_odds'] * 0.55) & (race_df['単勝_num'] >= 5.0)

    race_df['ev_brain2'] = (race_df['win_prob'] * race_df['単勝_num']).fillna(0).round(2)
    max_p = max(race_df['win_prob'].max(), 0.01)
    
    race_df['score_brain1'] = ((race_df['win_prob'] / max_p) * 75.0 + (race_df['ev_brain2'].clip(0, 3.0) / 3.0) * 20.0).fillna(10).clip(10, 99).astype(int)
    
    # 異常オッズ馬には強制的にスコアボーナス
    race_df.loc[race_df['is_abnormal'], 'score_brain1'] = (race_df['score_brain1'] + 15).clip(upper=99)

    return race_df.sort_values(by=['score_brain1', 'win_prob'], ascending=[False, False]).reset_index(drop=True)

st.sidebar.header("🔄 画面の更新")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
if st.sidebar.button("🔄 キャッシュ完全クリア＆リロード", use_container_width=True): st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

def get_mark(idx): return "◎ 本命" if idx == 0 else "◯ 対抗" if idx == 1 else "▲ 単穴" if idx == 2 else "△ 連下" if idx == 3 else "消"

def generate_beautiful_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>騎手</th><th>脚質</th><th>AIスコア</th><th>オッズ</th><th>適正オッズ</th><th>期待値</th><th>異常検知</th><th>タイム指</th><th>印</th></tr></thead><tbody>"
    for i, r in disp_df.iterrows():
        mark = get_mark(i)
        b_cls = "badge-honmei" if "◎" in mark else "badge-taikou" if "◯" in mark else "badge-tana" if "▲" in mark else "badge-renka" if "△" in mark else "badge-keshi"
        k_style = "background:#bdc3c7; color:#fff !important;"
        if r['脚質'] == "逃": k_style = "background:#ff7675; color:#fff !important;"
        elif r['脚質'] == "先": k_style = "background:#e67e22; color:#fff !important;"
        elif r['脚質'] == "差": k_style = "background:#3498db; color:#fff !important;"
        elif r['脚質'] == "追": k_style = "background:#2ecc71; color:#fff !important;"
        
        abnormal_badge = "<span class='badge-alert'>🚨大口検知</span>" if r.get('is_abnormal', False) else "-"
        ev_style = "color:#e74c3c !important; font-weight:900;" if r['ev_brain2'] >= 1.0 else "color:#5a3d46 !important;"
        
        html += f"<tr><td style='font-weight:bold; color:#c94a65 !important;'>{int(r['馬番_num']):02d}</td><td style='text-align:left; font-weight:800; color:#5a3d46 !important;'>{r.get('馬名', '-')}</td><td style='color:#666666 !important;'>{r.get('騎手', '-')}</td><td><span style='{k_style} padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{r['脚質']}</span></td><td style='color:#5a3d46 !important;'><b>{int(r['score_brain1'])}点</b></td><td style='color:#666666 !important;'>{r['単勝_num']:.1f}倍</td><td style='color:#bdc3c7 !important; font-size:0.9em;'>{r.get('expected_odds', 0)}倍</td><td style='{ev_style}'><b>{r['ev_brain2']:.2f}</b></td><td>{abnormal_badge}</td><td><span class='badge-idx'>{r.get('custom_time_index', 0)}</span></td><td><span class='badge-mark {b_cls}'>{mark}</span></td></tr>"
    html += "</tbody></table></div>"
    return html

tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想 (アルティメット)", "📈 地方実戦成績"])
with tab_forecast:
    if df_future.empty: st.warning("⚠️ 本日の出馬表データが存在しません。")
    else:
        st.markdown("<div class='section-header'>🎯 予想レースを選択</div>", unsafe_allow_html=True)
        dates = sorted(df_future['day_label'].unique())
        selected_date = st.radio("開催日", dates, horizontal=True, label_visibility="collapsed")
        day_df = df_future[df_future['day_label'] == selected_date]
        places = day_df['place_name'].unique()
        place_tabs = st.tabs([f"📍 {p}" for p in places])
        for p_idx, place in enumerate(places):
            with place_tabs[p_idx]:
                place_df = day_df[day_df['place_name'] == place]
                for i in range(0, len(place_df['r_num'].unique()), 6):
                    cols = st.columns(6)
                    for j, r in enumerate(sorted(place_df['r_num'].unique())[i:i+6]):
                        rid = place_df[place_df['r_num'] == r]['race_id'].iloc[0]
                        if cols[j].button(f"{r}R", key=f"btn_{rid}", use_container_width=True, type="primary" if st.session_state['selected_race_id'] == rid else "secondary"): st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id'] and not df_future.empty:
        st.markdown("---")
        target_id = str(st.session_state['selected_race_id'])
        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0]
        race_display_name = f"{info['place_name']} {info['r_num']}R 【{info.get('race_name', '')}】"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        selected_baba = st.radio("🌧️ 現在の馬場状態を選択してください", ["良", "稍重", "重", "不良"], horizontal=True, index=["良", "稍重", "重", "不良"].index(st.session_state['baba_status']))
        if selected_baba != st.session_state['baba_status']: st.session_state['baba_status'] = selected_baba; st.rerun()
        
        scored_df = calculate_race_scores(target_id, df_future, baba_status=st.session_state['baba_status'])
        if scored_df is not None:
            st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAIスコア (📍 {info['place_name']}のコースバイアス反映済)</div>", unsafe_allow_html=True)
            st.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Geminiで【コース適性・異常オッズを加味した厳選4頭】を予想", type="primary", use_container_width=True):
            if not api_key_input: st.error("【設定エラー】APIキーが見つかりません。"); st.stop()
            
            abnormal_horses = scored_df[scored_df['is_abnormal']]
            abnormal_info = f"⚠️ 本レースでは、AI適正オッズに対して異常に売れている【大口投票の疑いがある穴馬】が {len(abnormal_horses)} 頭います。" if not abnormal_horses.empty else ""
            
            table_summary = [f"印:{get_mark(idx)} | 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | 脚質:{row['脚質']} | オッズ:{row['単勝_num']}倍 (適正:{row.get('expected_odds',0)}倍) | 期待値:{row['ev_brain2']:.2f} | 異常投票:{'あり' if row.get('is_abnormal') else 'なし'}" for idx, row in scored_df.iterrows() if get_mark(idx) != "消"]

            sys_inst = f'''あなたは地方競馬の回収率・穴馬特化型AI「勝ち子ちゃん」です。
【究極の回収率ロジック】
* 競馬場は「{info['place_name']}」、馬場は「{st.session_state['baba_status']}」です。この競馬場特有のコースバイアス（有利な脚質）を必ず展開予想に組み込んでください。
* {abnormal_info} (異常オッズ馬がいる場合は、関係者の勝負気配とみなし積極的に▲や△に抜擢してください)
* 予想は上位4頭のみ厳守。ガチガチ決着は避け、必ず期待値1.0以上の馬や異常オッズ馬を絡めてヒモ荒れを狙ってください。
* 買い目は【4頭のみ・合計3〜4点】。
【出力フォーマット】
---
### 🌸 馬場・コース適性と異常オッズ考察
* （展開見解と穴馬激走理由）
### 🎯 勝ち子ちゃんの厳選4頭
* ◎ 本命: 〇〇番 - 理由
* ◯ 対抗: 〇〇番 - 理由
* ▲ 単穴: 〇〇番 - 理由
* △ 連下: 〇〇番 - 理由
### 🎀 推奨買い目
* 馬単/馬連: ◎→◯,▲,△(3点) 等
---'''
            with st.spinner("🎀 競馬場バイアスと異常オッズから、極秘の少数買い目を生成中..."):
                try:
                    res_text = genai.Client(api_key=api_key_input).models.generate_content(model='gemini-2.5-flash', contents=f"対象: {selected_date} {race_display_name}\n\n対象4頭:\n" + "\n".join(table_summary), config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.3)).text
                    st.markdown(f"<div class='gemini-output-box'>{res_text}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"エラー: {e}")