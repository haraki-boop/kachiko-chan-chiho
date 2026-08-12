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
st.set_page_config(page_title="AI予想 勝ち子ちゃん | 地方版", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fcf9f9; font-family: 'Helvetica Neue', 'Hiragino Kaku Gothic ProN', Arial, sans-serif; }
    
    h1 { font-size: 1.9rem !important; color: #c94a65 !important; font-weight: 800; }
    h2 { font-size: 1.4rem !important; color: #5a3d46 !important; }
    
    .section-header {
        font-size: 1.25rem; font-weight: 800; color: #c94a65;
        margin-top: 1.5rem; margin-bottom: 1rem;
        border-bottom: 2px solid #f2cdd5; padding-bottom: 6px;
    }
    
    .table-container {
        width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;
        margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
    
    .kachi-table {
        width: 100%; border-collapse: collapse; margin-bottom: 0; background-color: #ffffff; white-space: nowrap;
    }
    .kachi-table thead tr { background: linear-gradient(90deg, #d9788f, #e895a7); color: #ffffff; font-weight: bold; }
    .kachi-table th { padding: 10px 12px; text-align: center; border-right: 1px solid rgba(255,255,255,0.2); }
    .kachi-table th:last-child { border-right: none; }
    .kachi-table td { padding: 8px 12px; text-align: center; border-bottom: 1px solid #f2eced; color: #5a3d46; font-weight: 500; }
    .kachi-table tbody tr:hover td { background: #fff5f7; }
    
    .badge-mark { color: #fff; padding: 4px 10px; border-radius: 20px; font-weight: bold; font-size: 0.85em; display: inline-block; min-width: 55px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .badge-honmei { background: linear-gradient(135deg, #ff4757, #ff6b81); }
    .badge-taikou { background: linear-gradient(135deg, #3742fa, #5352ed); }
    .badge-tana   { background: linear-gradient(135deg, #2ed573, #7bed9f); }
    .badge-renka  { background: linear-gradient(135deg, #ffa502, #eccc68); color: #333; }
    .badge-keshi  { background: #e0e0e0; color: #7f8c8d; box-shadow: none; }
    
    .badge-idx { background-color: #fff0f3; color: #c94a65; font-weight: 800; padding: 3px 8px; border-radius: 6px; font-size: 0.85em; border: 1px solid #f2cdd5; }

    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
        .kachi-table { font-size: 12px; }
        .kachi-table th, .kachi-table td { padding: 5px 6px; }
        h1 { font-size: 1.4rem !important; } 
        h2 { font-size: 1.1rem !important; }
        .stButton button p { font-size: 0.7rem !important; white-space: nowrap !important; }
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1:
    st.write("🌸")
with col2:
    st.title("AI予想 勝ち子ちゃん (地方穴馬特化)")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

# ==========================================
# 🔑 2. 定数と環境設定・読み込み
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

HISTORY_CSV = "prediction_history_chiho.csv"
FUTURE_CSV = "future_races_chiho.csv"
ML_TARGET_CSV = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

NAR_PLACES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

def clean_horse_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[\s\u3000]+', '', unicodedata.normalize('NFKC', str(name)))

def load_csv_utf8(path, dtype_dict=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'euc-jp']:
        try:
            return pd.read_csv(path, dtype=dtype_dict, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        try: return joblib.load(MODEL_FILE)
        except: return None
    return None

def load_data_files():
    df_p = load_csv_utf8(ML_TARGET_CSV, dtype_dict={'race_id': str})
    if not df_p.empty and '馬名' in df_p.columns:
        df_p['馬名_clean'] = df_p['馬名'].astype(str).apply(clean_horse_name)
        
    df_f = load_csv_utf8(FUTURE_CSV, dtype_dict={'race_id': str, '馬番': str})
    if not df_f.empty and 'race_id' in df_f.columns:
        df_f['place_code'] = df_f['race_id'].str[4:6]
        df_f['place_name'] = df_f['place_code'].map(NAR_PLACES).fillna("地方")
        df_f['r_num'] = df_f['race_id'].str[10:12].astype(int)
        df_f['day_label'] = df_f['date'] if 'date' in df_f.columns else datetime.now().strftime("%Y-%m-%d")
        
    df_h = load_csv_utf8(HISTORY_CSV, dtype_dict={'race_id': str, 'honmei_umaban': str, 'partners': str})
    if not df_h.empty:
        if 'partners' not in df_h.columns: df_h['partners'] = ""
        for col in ['pay_tansho', 'pay_umaren', 'pay_wide', 'pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form']:
            if col not in df_h.columns: df_h[col] = 0
    else:
        df_h = pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay', 'pay_tansho', 'pay_umaren', 'pay_wide', 'pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form'])
        
    return df_p, df_f, df_h

model_data = load_model()
df_past, df_future, df_history = load_data_files()

@st.cache_data
def build_past_dicts(df_p):
    jockey_dict, horse_dict = {}, {}
    if not df_p.empty:
        if 'target_win' in df_p.columns:
            j_stats = df_p.groupby('騎手')['target_win'].agg(['count', 'mean'])
            for j, row in j_stats.iterrows():
                jockey_dict[j] = {'j_runs': row['count'], 'j_win_rate': row['mean']}
        
        past = df_p.sort_values(['馬名_clean', 'date']) if 'date' in df_p.columns else df_p
        for h, group in past.groupby('馬名_clean'):
            recent_3 = group.tail(3)
            avg_1c = recent_3['first_corner'].mean() if 'first_corner' in recent_3.columns else 8.0
            last_row = recent_3.iloc[-1]
            prev_dist = last_row['distance_num'] if 'distance_num' in recent_3.columns else 1400
            prev_weight = last_row['horse_weight'] if 'horse_weight' in recent_3.columns else 470
            avg_rank = recent_3['target_rank'].mean() if 'target_rank' in recent_3.columns else 6.0
            
            horse_dict[h] = {
                'prev_dist': prev_dist,
                'prev_1c': avg_1c,
                'prev_weight': prev_weight,
                'recent_avg_rank': avg_rank
            }
    return jockey_dict, horse_dict

jockey_dict, horse_dict = build_past_dicts(df_past)

def get_kyakushitsu(fc):
    """1コーナーの通過順位から脚質を判定"""
    if pd.isna(fc) or fc == 0: return "-"
    if fc <= 2.0: return "逃"
    elif fc <= 4.5: return "先"
    elif fc <= 7.5: return "差"
    else: return "追"

# ==========================================
# 3. AIスコア ＆ 穴馬・高期待値ロジック算出
# ==========================================
def calculate_race_scores(race_id_target, target_df):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    model = model_data.get('model') if model_data else None
    features = model_data.get('features', []) if model_data else []

    race_df['単勝_num'] = pd.to_numeric(race_df.get('単勝'), errors='coerce').fillna(15.0)
    race_df['人気_num'] = pd.to_numeric(race_df.get('人気'), errors='coerce').fillna(99.0)
    race_df['distance_num'] = pd.to_numeric(race_df.get('distance'), errors='coerce').fillna(1400)
    race_df['weight_num'] = pd.to_numeric(race_df.get('斤量'), errors='coerce').fillna(54.0)
    race_df['gate_num'] = pd.to_numeric(race_df.get('馬番'), errors='coerce').fillna(8.0)
    race_df['馬番_num'] = pd.to_numeric(race_df.get('馬番'), errors='coerce').fillna(0)
    race_df['馬名_clean'] = race_df['馬名'].astype(str).apply(clean_horse_name)

    def get_h(name, k, default): return horse_dict.get(name, {}).get(k, default)
    def get_j(name, k, default): return jockey_dict.get(name, {}).get(k, default)
    
    race_df['prev_dist'] = race_df['馬名_clean'].apply(lambda x: get_h(x, 'prev_dist', 1400))
    race_df['dist_change'] = race_df['distance_num'] - race_df['prev_dist']
    race_df['first_corner'] = race_df['馬名_clean'].apply(lambda x: get_h(x, 'prev_1c', 6))
    race_df['recent_avg_rank'] = race_df['馬名_clean'].apply(lambda x: get_h(x, 'recent_avg_rank', 6.0))
    race_df['horse_weight'] = race_df['馬名_clean'].apply(lambda x: get_h(x, 'prev_weight', 470))
    
    race_df['jockey_win_rate'] = race_df['騎手'].apply(lambda x: get_j(x, 'j_win_rate', 0.08))

    # 脚質の付与
    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)

    # ⏱️ タイム指数（30〜99点の範囲内にガード）
    raw_time = 75.0 - (race_df['recent_avg_rank'].clip(lower=1.0, upper=14.0) - 3.0) * 3.0 + (race_df['weight_num'] - 54.0) * 1.5
    race_df['custom_time_index'] = raw_time.clip(30.0, 99.0).round(1)

    # 🚀 スタート指数（30〜99点の範囲内にガード）
    raw_start = (12.0 - race_df['first_corner'].clip(upper=10.0)) * 6.5 + race_df['gate_num'] * 0.5
    race_df['custom_start_index'] = raw_start.clip(30.0, 99.0).round(1)

    # 🌟 勝率の算出
    inv_odds = 1.0 / race_df['単勝_num'].clip(lower=1.0)
    base_prob = np.power(inv_odds, 1.2)
    
    if model:
        try:
            X = race_df[features].fillna(0.0)
            preds = model.predict(X)
            preds_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
            base_prob = base_prob * 0.6 + np.power(preds_norm + 0.1, 1.5) * 0.4
        except: pass

    time_rank_norm = (race_df['custom_time_index'] - race_df['custom_time_index'].min()) / (race_df['custom_time_index'].max() - race_df['custom_time_index'].min() + 1e-5)
    start_rank_norm = (race_df['custom_start_index'] - race_df['custom_start_index'].min()) / (race_df['custom_start_index'].max() - race_df['custom_start_index'].min() + 1e-5)
    
    ana_bonus = (time_rank_norm * 0.5 + start_rank_norm * 0.5) * (race_df['単勝_num'] >= 10.0).astype(int) * 0.05
    
    race_df['win_prob'] = base_prob + ana_bonus
    race_df['win_prob'] = race_df['win_prob'] / race_df['win_prob'].sum()

    # 期待値（EV）
    race_df['ev_brain2'] = (race_df['win_prob'] * race_df['単勝_num']).round(2)

    # AIスコア
    max_p = race_df['win_prob'].max()
    ev_score = (race_df['ev_brain2'].clip(0, 3.0) / 3.0) * 20.0
    prob_score = (race_df['win_prob'] / max_p) * 75.0
    race_df['score_brain1'] = (prob_score + ev_score).clip(10, 98).round().astype(int)

    return race_df.sort_values(by=['score_brain1', 'win_prob'], ascending=[False, False]).reset_index(drop=True)

def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and not sdf.empty:
            top_contenders = len(sdf[sdf['score_brain1'] >= 80])
            if top_contenders == 3:
                markers[rid] = "🔥激熱"
            elif top_contenders == 2:
                markers[rid] = "💥熱"
            else:
                markers[rid] = ""
    return markers

markers = get_all_markers()

# ==========================================
# 4. サイドバー UI & テーブル生成
# ==========================================
st.sidebar.header("🔄 画面の更新")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
if st.sidebar.button("🔄 キャッシュ完全クリア＆リロード", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

def get_mark(idx):
    if idx == 0: return "◎ 本命"
    if idx == 1: return "◯ 対抗"
    if idx == 2: return "▲ 単穴"
    if idx == 3: return "△ 連下"
    return "消"

def generate_beautiful_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>騎手</th><th>脚質</th><th>AIスコア</th><th>勝率</th><th>オッズ</th><th>期待値</th><th>タイム指</th><th>スタート指</th><th>印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        ev_val, odds_val = float(r.get('ev_brain2', 0)), float(r.get('単勝_num', 0))
        mark = get_mark(i)
        
        b_cls = "badge-keshi"
        if "◎" in mark: b_cls = "badge-honmei"
        elif "◯" in mark: b_cls = "badge-taikou"
        elif "▲" in mark: b_cls = "badge-tana"
        elif "△" in mark: b_cls = "badge-renka"
        
        ev_style = "color:#e74c3c; font-weight:900;" if ev_val >= 1.0 else "color:#5a3d46;"
        
        kyaku = r.get('脚質', '-')
        k_style = "background:#bdc3c7; color:#fff;"
        if kyaku == "逃": k_style = "background:#ff7675; color:#fff;"
        elif kyaku == "先": k_style = "background:#e67e22; color:#fff;"
        elif kyaku == "差": k_style = "background:#3498db; color:#fff;"
        elif kyaku == "追": k_style = "background:#2ecc71; color:#fff;"
        kyaku_badge = f"<span style='{k_style} padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{kyaku}</span>"
        
        html += f"""<tr>
<td style='font-weight:bold; color:#c94a65;'>{int(r['馬番_num']):02d}</td>
<td style='text-align:left; font-weight:800; color:#5a3d46;'>{r.get('馬名', '-')}</td>
<td style='color:#7f8c8d;'>{r.get('騎手', '-')}</td>
<td>{kyaku_badge}</td>
<td style='color:#5a3d46;'><b>{int(r['score_brain1'])}点</b></td>
<td style='color:#c94a65;'><b>{float(r.get('win_prob',0))*100:.1f}%</b></td>
<td style='color:#7f8c8d;'>{odds_val:.1f}倍</td>
<td style='{ev_style}'><b>{ev_val:.2f}</b></td>
<td><span class='badge-idx'>{r.get('custom_time_index', 0)}</span></td>
<td>{r.get('custom_start_index', 0)}</td>
<td><span class='badge-mark {b_cls}'>{mark}</span></td>
</tr>"""
    html += "</tbody></table></div>"
    return html

# ==========================================
# 5. メインUI (レース予想画面)
# ==========================================
tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想 (勝ち子ちゃん)", "📈 地方実戦成績"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 本日の出馬表データが存在しません。`scrape_chiho_today.py` を実行してください。")
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
                races = sorted(place_df['r_num'].unique())
                
                for i in range(0, len(races), 6):
                    chunk = races[i:i+6]
                    cols = st.columns(6)
                    for j, r in enumerate(chunk):
                        col = cols[j]
                        rid = place_df[place_df['r_num'] == r]['race_id'].iloc[0]
                        mark = markers.get(rid, "")
                        label = f"{r}R {mark}".strip()
                        btn_type = "primary" if "熱" in mark else "secondary"
                        if col.button(label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
                            st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id'] and not df_future.empty:
        st.markdown("---")
        target_id = str(st.session_state['selected_race_id'])
        target_rows = df_future[df_future['race_id'].astype(str) == target_id]
        
        if target_rows.empty:
            st.session_state['selected_race_id'] = None
            st.rerun()
            
        info = target_rows.iloc[0]
        rname = info.get('race_name', "")
        race_display_name = f"{info['place_name']} {info['r_num']}R 【{rname}】"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        scored_df = calculate_race_scores(target_id, df_future)
        
        if scored_df is not None:
            st.markdown("<div class='section-header'>📊 勝ち子ちゃんのAIスコア＆独自指数</div>", unsafe_allow_html=True)
            st.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎀 Geminiで【穴馬を含めた厳選4頭】を予想", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
            if scored_df is None or len(scored_df) < 4:
                st.error("出走頭数が少ないため予想をスキップします。")
                st.stop()

            high_ev_horses = scored_df[scored_df['ev_brain2'] >= 1.0]
            ana_info = f"本レースで期待値(EV)が1.0を超える激走候補の穴馬は、{len(high_ev_horses)}頭です。"

            table_summary = []
            for idx, row in scored_df.iterrows():
                mark = get_mark(idx)
                if mark != "消":
                    table_summary.append(
                        f"印:{mark} | 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | 脚質:{row['脚質']} | 騎手:{row['騎手']} | "
                        f"オッズ:{row['単勝_num']}倍 | 期待値:{row['ev_brain2']:.2f} | AIスコア:{row['score_brain1']}点 | "
                        f"タイム指:{row['custom_time_index']} | スタート指:{row['custom_start_index']}"
                    )

            system_instruction = f"""
あなたは地方競馬の回収率・穴馬特化型AI「勝ち子ちゃん」です。

【回収率を跳ねさせる穴馬戦略】
* {ana_info}
* 予想は提供された**【◎, ◯, ▲, △】の上位4頭のみ**を厳守してください。
* 各馬の「脚質」を確認し、「逃げ・先行馬が多数いるなら差し馬が有利になる」などの展開予想を必ず組み込んでください。
* ガチガチの人気決着（例: 1番人気→2番人気）はトリガミになるため、必ず「期待値（EV）が1.0を超えている馬」や「展開が向きそうな人気薄」を▲や△に抜擢し、ヒモ荒れを狙ってください。
* 買い目は【4頭のみ・合計3〜4点】の極小点数で構成し、回収率を最大化してください。

【出力フォーマット】
---
### 🌸 レース展開と穴馬の狙い目
* （脚質分布に基づくペース想定と、▲・△に抜擢した穴馬の激走理由を解説）

### 🎯 勝ち子ちゃんの厳選4頭
* **◎ 本命:** 〇〇番（馬名） - （脚質・抜擢理由）
* **◯ 対抗:** 〇〇番（馬名） - （見解）
* **▲ 単穴:** 〇〇番（馬名） - （期待値や展開利などのヒモ穴根拠）
* **△ 連下:** 〇〇番（馬名） - （押さえの根拠）

### 🎀 推奨買い目（4頭のみ・合計3〜4点）
* **馬単 / 馬連:** ◎ → ◯, ▲, △ (3点)
* **3連単 (1着固定):** ◎ → ◯, ▲ → ◯, ▲, △ (4点)
* **ワイド:** ◎ - ▲, △ (2点)
---
"""
            prompt = f"対象レース: {selected_date} {race_display_name}\n\n対象4頭データ:\n" + "\n".join(table_summary)

            with st.spinner("🎀 指数と期待値から、ヒモ荒れを捉える少数買い目を生成中..."):
                client = genai.Client(api_key=api_key_input)
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.3
                        )
                    )
                    res_text = response.text
                    st.markdown(f"<div style='background:white; padding:24px; border-radius:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); border: 2px solid #f2cdd5;'>{res_text}</div>", unsafe_allow_html=True)
                    
                    honmei_match = re.search(r'◎.*?[）:]\s*(\d+)番', res_text)
                    h_umaban = int(honmei_match.group(1)) if honmei_match else int(scored_df.iloc[0]['馬番_num'])
                    all_nums = re.findall(r'(\d+)番', res_text)
                    partners_str = ",".join(list(dict.fromkeys([n for n in all_nums if int(n) != h_umaban]))[:3])
                    
                    if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                        new_record = pd.DataFrame([{
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'race_id': str(target_id),
                            'race_name': race_display_name,
                            'honmei_umaban': h_umaban,
                            'partners': partners_str,
                            'honmei_name': "履歴",
                            'result_pay': "",
                            'pay_tansho': 0, 'pay_umaren': 0, 'pay_wide': 0,
                            'pay_sanrenpuku': 0, 'pay_sanrentan_axis': 0, 'pay_sanrentan_form': 0
                        }])
                        df_history = pd.concat([df_history, new_record], ignore_index=True)
                        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
                    st.success("📝 予想を実戦履歴に記録しました！")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

# ==========================================
# 6. ダッシュボード
# ==========================================
with tab_dashboard:
    st.markdown("<div class='section-header'>📈 地方実戦成績ダッシュボード</div>", unsafe_allow_html=True)
    if df_history.empty: 
        st.info("まだ地方予想履歴がありません。")
    else:
        df_history['result_pay'] = df_history['result_pay'].replace(['None', 'nan', 'NaN', ''], np.nan)
        df_history['datetime'] = pd.to_datetime(df_history['date'], errors='coerce')
        df_history['year_month'] = df_history['datetime'].dt.strftime('%Y年%m月')
        df_history['just_date'] = df_history['datetime'].dt.strftime('%Y-%m-%d')
        
        tab_total, tab_month, tab_day = st.tabs(["🏆 総合成績", "📅 月別成績", "📆 日別成績"])
        
        def render_dashboard_for_df(raw_df, title_prefix):
            total_races = len(raw_df)
            if total_races == 0:
                st.info(f"この期間のレースはありません。")
                return
            
            finished_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').notna()]
            pending_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').isna()]
            total = len(finished_df)
            
            if total == 0:
                st.markdown(f"**{title_prefix} 確定レース**: 0 件 （結果待ち: {total_races} 件）")
            else:
                hits = len(finished_df[pd.to_numeric(finished_df['result_pay'], errors='coerce') > 0])
                returns = pd.to_numeric(finished_df['result_pay'], errors='coerce').sum()
                
                invested_total, inv_tansho, inv_umaren, inv_wide, inv_sanrenpuku, inv_sanrentan_axis, inv_sanrentan_form = 0, 0, 0, 0, 0, 0, 0
                
                for _, r in finished_df.iterrows():
                    p_list = [x for x in str(r.get('partners', '')).split(',') if x.strip().isdigit()]
                    p_len = len(p_list)
                    if p_len > 0:
                        inv_tansho += 100
                        inv_umaren += p_len * 100
                        inv_wide += p_len * 100
                        box_count = p_len + 1
                        if box_count >= 3: inv_sanrenpuku += int(box_count * (box_count - 1) * (box_count - 2) / 6) * 100
                        if p_len >= 2:
                            inv_sanrentan_axis += (p_len * (p_len - 1)) * 100
                            inv_sanrentan_form += (2 * (p_len - 1)) * 100
                
                invested_total = inv_tansho + inv_umaren + inv_wide + inv_sanrenpuku + inv_sanrentan_axis
                roi_total = (returns / invested_total) * 100 if invested_total > 0 else 0.0
                profit_total = int(returns - invested_total)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🎯 的中率", f"{(hits/total)*100:.1f}%", f"{hits} / {total} レース的中", delta_color="off")
                col2.metric("💰 回収率", f"{roi_total:.1f}%", delta_color="normal" if profit_total >= 0 else "inverse")
                col3.metric("💴 収支", f"{profit_total:,} 円")
                
                st.markdown("<br><h5 style='color:#c94a65;'>🎫 券種別の詳細データ</h5>", unsafe_allow_html=True)
                
                def make_ticket_card(col, name, hits_val, returns_val, inv_val):
                    roi_val = (returns_val / inv_val) * 100 if inv_val > 0 else 0
                    profit_val = int(returns_val - inv_val)
                    color = "#c94a65" if profit_val < 0 else "#2ecc71"
                    sign = "+" if profit_val > 0 else ""
                    col.markdown(f'''
                    <div style="border:1px solid #f2cdd5; padding:10px; border-radius:12px; text-align:center; background-color:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                        <div style="font-weight:bold; font-size:1.05em; color:#5a3d46; margin-bottom:5px;">{name}</div>
                        <div style="font-size:0.8em; color:#7f8c8d;">投資: {int(inv_val):,}円</div>
                        <div style="font-size:0.8em; color:#7f8c8d;">的中: {(hits_val/total)*100:.1f}%</div>
                        <div style="font-size:0.8em; color:#7f8c8d;">回収: {roi_val:.1f}%</div>
                        <div style="font-weight:bold; color:{color}; margin-top:5px; font-size:1.1em;">{sign}{profit_val:,} 円</div>
                    </div>
                    ''', unsafe_allow_html=True)

                t_cols1 = st.columns(3)
                t_cols2 = st.columns(3)
                if 'pay_tansho' in finished_df.columns:
                    make_ticket_card(t_cols1[0], "単勝", len(finished_df[pd.to_numeric(finished_df['pay_tansho'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_tansho'], errors='coerce').sum(), inv_tansho)
                    make_ticket_card(t_cols1[1], "馬連", len(finished_df[pd.to_numeric(finished_df['pay_umaren'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_umaren'], errors='coerce').sum(), inv_umaren)
                    make_ticket_card(t_cols1[2], "ワイド", len(finished_df[pd.to_numeric(finished_df['pay_wide'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_wide'], errors='coerce').sum(), inv_wide)
                    make_ticket_card(t_cols2[0], "三連複", len(finished_df[pd.to_numeric(finished_df['pay_sanrenpuku'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_sanrenpuku'], errors='coerce').sum(), inv_sanrenpuku)
                    make_ticket_card(t_cols2[1], "三連単 (1着固定)", len(finished_df[pd.to_numeric(finished_df['pay_sanrentan_axis'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_sanrentan_axis'], errors='coerce').sum(), inv_sanrentan_axis)
                    make_ticket_card(t_cols2[2], "三連単 (F)", len(finished_df[pd.to_numeric(finished_df['pay_sanrentan_form'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_sanrentan_form'], errors='coerce').sum(), inv_sanrentan_form)

        with tab_total: render_dashboard_for_df(df_history, "総合")