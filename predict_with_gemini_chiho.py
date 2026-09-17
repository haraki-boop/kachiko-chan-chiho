import os
import re
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime, timezone, timedelta
from google import genai
from google.genai import types

# ==========================================
# 🎨 アプリの基本設定 & スタイル定義
# ==========================================
st.set_page_config(page_title="AI予想 勝ち子ちゃん | 馬場補正・超高速版", page_icon="🌸", layout="wide")

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
    .gemini-output-box { background-color: #ffffff !important; color: #222222 !important; padding: 20px; border-radius: 12px; border: 2px solid #f2cdd5; margin-top: 15px; }
    .note-icon { font-size: 1.1em; margin-right: 3px; }
    .missing-data { font-size: 0.85em; color: #999; background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .track-bias-badge { background: #3498db; color: #fff; padding: 3px 6px; border-radius: 6px; font-size: 0.8em; font-weight: bold; margin-top: 2px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (馬場補正搭載版)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'gemini_results' not in st.session_state: st.session_state['gemini_results'] = {}

def set_race_id(rid): st.session_state['selected_race_id'] = str(rid)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FUTURE_CSV = "future_races_chiho.csv"
CACHE_FILE = "app_cache_chiho.pkl"
NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

@st.cache_data
def load_future_data():
    if os.path.exists(FUTURE_CSV):
        for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
            try:
                df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, low_memory=False, encoding=enc)
                if not df.empty:
                    df['race_id'] = pd.to_numeric(df['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)
                    df['place_code'] = df['race_id'].astype(str).str[4:6]
                    df['place_name'] = df['place_code'].map(NAR_PLACES).fillna("地方")
                    df['r_num'] = pd.to_numeric(df['race_id'].str[10:12], errors='coerce').fillna(1).astype(int)
                    df['day_label'] = df['date'].astype(str).str.strip() if 'date' in df.columns else "当日"
                    return df
            except Exception: continue
    return pd.DataFrame()

@st.cache_resource
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return joblib.load(CACHE_FILE)
        except Exception as e:
            st.error(f"⚠️ キャッシュ読み込みエラー: {e}")
            return {}
    else:
        st.error(f"⚠️ キャッシュファイル '{CACHE_FILE}' が見つかりません。")
        return {}

df_future = load_future_data()
cache_data = load_cache()

# ⚙️ サイドバー：設定および手動馬場状態の選択
st.sidebar.header("⚙️ 当日の状況・設定")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")

st.sidebar.markdown("---")
st.sidebar.subheader("☔ 当日の馬場状態選択")
track_condition = st.sidebar.radio(
    "馬場状態を指定してください",
    options=["良", "稍重", "重", "不良"],
    index=0
)

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
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>特注 / 馬場補正</th><th>騎手(勝率)</th><th>脚質</th><th>人気(オッズ)</th><th>連対率</th><th>指数実績<br>(タイム/ダッシュ)</th><th>AI偏差値</th><th>AI印</th><th>Gemini印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        ai_mark = get_mark(i)
        b_cls_ai = "badge-honmei" if "◎" in ai_mark else "badge-taikou" if "◯" in ai_mark else "badge-tana" if "▲" in ai_mark else "badge-renka" if "△" in ai_mark else "badge-tana" if "☆" in ai_mark else "badge-keshi"
        
        gem_mark = r.get('gemini_mark', '-')
        if gem_mark == "-":
            gem_str = "<span style='color:#ccc; font-weight:bold;'>-</span>"
        else:
            b_cls_gem = "badge-honmei" if "◎" in gem_mark else "badge-taikou" if "◯" in gem_mark else "badge-tana" if "▲" in gem_mark else "badge-renka" if "△" in gem_mark else "badge-tana" if "☆" in gem_mark else "badge-keshi"
            gem_str = f"<span class='badge-mark {b_cls_gem}'>{gem_mark}</span>"

        kyaku = r.get('脚質', '-')
        k_style = "background:#ff7675;" if kyaku == "逃" else "background:#e67e22;" if kyaku == "先" else "background:#3498db;" if kyaku == "差" else "background:#2ecc71;"

        jockey_name = str(r.get('騎手', r.get('jockey_name', r.get('騎手_clean', '-')))).strip()
        j_win = float(r.get('jockey_win_rate', 0.0)) * 100
        jockey_str = f"{jockey_name}<br><span style='font-size:0.8em; color:#666;'>({j_win:.1f}%)</span>"
        
        pop_val = r.get('人気', r.get('popularity', '-'))
        odds_val = r.get('オッズ', r.get('odds', '-'))
        try:
            pop_str = f"<b>{int(float(pop_val))}</b>人気<br><span style='font-size:0.8em; color:#666;'>({float(odds_val):.1f}倍)</span>"
        except:
            pop_str = "<span style='color:#ccc;'>-</span>"

        t_idx_val = r.get('eff_my_time_idx', r.get('prev_my_time_idx', np.nan))
        s_idx_val = r.get('eff_my_start_idx', r.get('custom_start_index', np.nan))
        
        is_missing = False
        try:
            t_f, s_f = float(t_idx_val), float(s_idx_val)
            if pd.isna(t_f) or pd.isna(s_f) or t_f == 0.0:
                is_missing = True
        except:
            is_missing = True

        if is_missing:
            idx_str = "<span class='missing-data'>データ無</span>"
            t_idx = 0
        else:
            t_idx = int(t_f)
            s_idx = int(s_f)
            idx_str = f"<span style='font-size:0.85em;'><b>{t_idx}</b> / <span style='color:#e67e22;'><b>{s_idx}</b></span></span>"

        notes = []
        if not is_missing and t_idx >= 105: notes.append("<span class='note-icon'>⚡</span>時計上位")
        
        track_bonus = float(r.get('track_bonus', 0.0))
        if track_bonus > 0:
            notes.append(f"<span class='track-bias-badge'>☔ 重適性 +{track_bonus:.1f}</span>")
        elif track_bonus < 0:
            notes.append(f"<span class='track-bias-badge' style='background:#95a5a6;'>☀ 良適性</span>")

        note_str = "<br>".join(notes) if notes else "<span style='color:#ccc;'>-</span>"
        
        u_num_val = r.get('馬番_num', r.get('馬番', r.get('gate_num', 0)))
        u_num = int(float(u_num_val)) if pd.notna(u_num_val) else 0
        h_name = str(r.get('馬名', r.get('馬名_clean', '-'))).strip()
        
        raw_score = r.get('score_disp', '-')
        score_str = f"<b>{raw_score}</b>" if raw_score != "-" else "-"
        rentai = int(float(r.get('jockey_rentai_rate', 0.0)) * 100) if pd.notna(r.get('jockey_rentai_rate', 0.0)) else 0

        html += f"""<tr>
<td style='font-weight:bold; color:#c94a65 !important;'>{u_num:02d}</td>
<td style='text-align:left; font-weight:800; color:#5a3d46 !important;'>{h_name}</td>
<td style='font-size:0.85em; font-weight:bold; color:#e67e22;'>{note_str}</td>
<td style='color:#666666 !important;'>{jockey_str}</td>
<td><span style='{k_style} color:#fff !important; padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{kyaku}</span></td>
<td>{pop_str}</td>
<td style='color:#5a3d46 !important;'><b>{rentai}%</b></td>
<td>{idx_str}</td>
<td style='color:#5a3d46 !important; font-size:1.1em;'>{score_str}</td>
<td><span class='badge-mark {b_cls_ai}'>{ai_mark}</span></td>
<td>{gem_str}</td>
</tr>"""
    html += "</tbody></table></div>"
    return html

if df_future.empty: 
    st.warning("⚠️ 出馬表データ (future_races_chiho.csv) が存在しないか空です。")
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
                    rid = str(place_df[place_df['r_num'] == r]['race_id'].iloc[0])
                    cols[j].button(
                        f"{r}R", 
                        key=f"btn_{rid}", 
                        use_container_width=True, 
                        type="primary" if st.session_state['selected_race_id'] == rid else "secondary",
                        on_click=set_race_id, args=(rid,)
                    )

if st.session_state['selected_race_id']:
    target_id = st.session_state['selected_race_id']
    st.markdown("---")
    
    if target_id not in cache_data:
        st.error(f"⚠️ このレース（{target_id}）の予測キャッシュが見つかりません。`update_all.py` を実行してキャッシュを更新してください。")
    else:
        scored_df = pd.DataFrame(cache_data[target_id])
        
        # ☔ 動的馬場バイアス補正
        scored_df['track_bonus'] = 0.0
        if track_condition in ["重", "不良"]:
            if 'heavy_track_place_rate' in scored_df.columns:
                rate = pd.to_numeric(scored_df['heavy_track_place_rate'], errors='coerce').fillna(0.0)
                scored_df['track_bonus'] = (rate * 5.0).clip(0, 5)
            else:
                kyaku_types = scored_df.get('脚質', pd.Series([''] * len(scored_df)))
                scored_df.loc[kyaku_types.isin(['逃', '先']), 'track_bonus'] = 1.5
        elif track_condition == "良":
            if 'heavy_track_place_rate' in scored_df.columns:
                rate = pd.to_numeric(scored_df['heavy_track_place_rate'], errors='coerce').fillna(0.0)
                scored_df['track_bonus'] = (rate * -2.0).clip(-2, 0)
        
        # スコア補正と再ソート
        if 'score_disp' in scored_df.columns:
            valid_scores = pd.to_numeric(scored_df['score_disp'], errors='coerce')
            scored_df['score_disp_base'] = valid_scores
            scored_df['score_disp'] = np.where(
                valid_scores.notna(),
                (valid_scores + scored_df['track_bonus']).round(1),
                "-"
            )
            scored_df = scored_df.sort_values(
                by=['score_disp_base'], ascending=False, na_position='last'
            ).reset_index(drop=True)
        
        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0] if not df_future.empty else {'place_name': '地方', 'r_num': '?', 'race_name': ''}
        race_display_name = f"{info['place_name']} {info['r_num']}R 【{info.get('race_name', '')}】 (指定馬場: {track_condition})"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)

        front_runners_count = len(scored_df[scored_df.get('脚質', '') == '逃']) + len(scored_df[scored_df.get('脚質', '') == '先'])
        pace_text = f"<br>🔥 <b>展開予想:</b> このレースは逃げ・先行馬が {front_runners_count} 頭います。{'ハイペース崩れに注意！差し馬の評価を上げています。' if front_runners_count >= 4 else 'ペースは落ち着きそうです。前残り注意。'}"

        def get_u_num(df, index):
            if len(df) > index:
                val = df.iloc[index].get('馬番_num', df.iloc[index].get('馬番', df.iloc[index].get('gate_num', 0)))
                return int(float(val)) if pd.notna(val) else 0
            return 0

        u_1 = get_u_num(scored_df, 0)
        u_2 = get_u_num(scored_df, 1)
        u_3 = get_u_num(scored_df, 2)
        u_4 = get_u_num(scored_df, 3)
        u_5 = get_u_num(scored_df, 4)

        if len(scored_df) > 1 and pd.notna(scored_df.iloc[0].get('score_disp_base')) and pd.notna(scored_df.iloc[1].get('score_disp_base')):
            s1 = float(scored_df.iloc[0].get('score_disp_base'))
            s2 = float(scored_df.iloc[1].get('score_disp_base'))
            score_diff = s1 - s2
        else:
            score_diff = 0

        if score_diff >= 4:
            rec_pattern_name = "🎯 【絶対能力上位・1着固定流し】 1位 ➔ 2〜4位 (計6点)"
            rec_text = f"1位の強さが抜けている（偏差値 {score_diff:.1f} 差）ため、頭固定の3連単で狙います。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}"
        else:
            rec_pattern_name = "🛡️ 【能力混戦・1頭軸流し】 1位 ➔ 2〜5位 (計6点)"
            rec_text = f"上位陣が能力拮抗（偏差値 {score_diff:.1f} 差）しているため、1位軸の3連複で広く狙います。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}, {u_5:02d}"

        st.markdown(f"""
        <div class='rec-banner-formation'>
            {rec_pattern_name}<br>
            <span style='font-size:0.85em; font-weight:normal;'>
            * <b>軸馬(1頭):</b> <b>{axis_horse}</b><br>
            * <b>相手(ヒモ):</b> {target_horses}<br>
            * <b>理由:</b> 🤖 <b>LambdaMARTの絶対偏差値</b>に基づく選定です。{rec_text}{pace_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAI評価 (📍 指定馬場「{track_condition}」補正版)</div>", unsafe_allow_html=True)

        scored_df['gemini_mark'] = "-"
        if target_id in st.session_state['gemini_results']:
            saved_marks = st.session_state['gemini_results'][target_id].get('marks', {})
            for h_num, g_mark in saved_marks.items():
                scored_df.loc[pd.to_numeric(scored_df.get('馬番_num', scored_df.get('馬番', scored_df.get('gate_num'))), errors='coerce') == float(h_num), 'gemini_mark'] = g_mark

        table_placeholder = st.empty()
        table_placeholder.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Gemini独自の完全独立予想を生成", use_container_width=True):
            if not api_key_input: 
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            table_summary = []
            neutral_df = scored_df.copy()
            neutral_df['u_num_temp'] = pd.to_numeric(neutral_df.get('馬番_num', neutral_df.get('馬番', neutral_df.get('gate_num', 0))), errors='coerce').fillna(99).astype(int)
            neutral_df = neutral_df.sort_values('u_num_temp')

            for idx, row in neutral_df.iterrows():
                u_n = row['u_num_temp']
                if u_n == 99: continue
                
                odds = row.get('オッズ', row.get('odds', '不明'))
                pop = row.get('人気', row.get('popularity', '不明'))
                ai_score = row.get('score_disp', '-')
                
                table_summary.append(
                    f"馬番:{u_n:02d} | 馬名:{row.get('馬名', row.get('馬名_clean', ''))} | 脚質:{row.get('脚質', '')} | 騎手:{row.get('騎手', row.get('騎手_clean', ''))} | AI偏差値(馬場補正込):{ai_score} | 人気/オッズ:{pop}人気({odds}倍)"
                )

            race_distance = info.get('distance', '不明')
            
            sys_inst = f"""あなたは地方競馬の事情通であり、AIデータと競馬のセオリーを融合させる天才予想家「勝ち子ちゃん（Gemini）」です。
当日の設定馬場は【{track_condition}】です。
AI偏差値、脚質、馬場状態【{track_condition}】を加味して最終印（◎, ◯, ▲, △, ☆）を打ってください。

◎ [馬番] 馬名 （理由）
◯ [馬番] 馬名 （理由）
▲ [馬番] 馬名 （理由）
△ [馬番] 馬名 （理由）
☆ [馬番] 馬名 （理由）
"""
            with st.spinner("🎀 Geminiが思考中..."):
                try:
                    client = genai.Client(api_key=api_key_input)
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"対象レース: {race_display_name} (距離: {race_distance}m)\n\n対象馬データ:\n" + "\n".join(table_summary),
                        config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.7) 
                    )
                    resp_text = response.text
                    
                    new_marks = {}
                    for line in resp_text.split('\n'):
                        match = re.search(r'([◎◯▲△☆]).*?\[(\d{1,2})\]', line)
                        if match:
                            mark = match.group(1)
                            horse_num = int(match.group(2))
                            new_marks[horse_num] = mark
                            
                    st.session_state['gemini_results'][target_id] = {
                        'marks': new_marks,
                        'text': resp_text
                    }
                    
                    for h_num, g_mark in new_marks.items():
                        scored_df.loc[pd.to_numeric(scored_df.get('馬番_num', scored_df.get('馬番', scored_df.get('gate_num'))), errors='coerce') == float(h_num), 'gemini_mark'] = g_mark
                    table_placeholder.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)
                    
                except Exception as e: st.error(f"エラー: {e}")

        if target_id in st.session_state['gemini_results']:
            clean_text_disp = re.sub(r'^[#\-\s]+', '', st.session_state['gemini_results'][target_id]['text'].strip())
            st.markdown(f"<div class='gemini-output-box'>{clean_text_disp}</div>", unsafe_allow_html=True)