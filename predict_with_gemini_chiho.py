import os
import re
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="AI予想 勝ち子ちゃん | 馬場補正・超高速版", page_icon="🌸", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fcf9f9 !important; color: #333333 !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    p, span, label, div, li, td, th { color: #333333; }
    h1 { font-size: 1.9rem !important; color: #c94a65 !important; font-weight: 800; }
    h2 { font-size: 1.4rem !important; color: #5a3d46 !important; }
    .section-header { font-size: 1.25rem; font-weight: 800; color: #c94a65 !important; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #f2cdd5; padding-bottom: 6px; }
    .rec-banner-formation { background: linear-gradient(135deg, #f39c12, #e67e22); color: #ffffff !important; padding: 18px 24px; border-radius: 12px; font-size: 1.3rem; font-weight: 900; box-shadow: 0 4px 15px rgba(243, 156, 18, 0.3); margin-bottom: 25px; border: 2px solid #d35400; }
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
    .missing-data { font-size: 0.85em; color: #999; background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .track-bias-badge { background: #3498db; color: #fff; padding: 3px 6px; border-radius: 6px; font-size: 0.8em; font-weight: bold; margin-top: 2px; display: inline-block; }
    .warning-banner { background: #e74c3c; color: #ffffff !important; padding: 16px 20px; border-radius: 10px; font-size: 1.1rem; font-weight: bold; margin-bottom: 20px; border: 2px solid #c0392b; }
</style>
""", unsafe_allow_html=True)

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
        try: return joblib.load(CACHE_FILE)
        except Exception: return {}
    return {}

df_future = load_future_data()
cache_data = load_cache()

st.sidebar.header("⚙️ 当日の状況・設定")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
track_condition = st.sidebar.radio("馬場状態を指定してください", options=["良", "稍重", "重", "不良"], index=0)

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

def generate_beautiful_table(disp_df, cond):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>馬場適性補正</th><th>騎手(勝率)</th><th>脚質</th><th>人気(オッズ)</th><th>連対率</th><th>指数実績<br>(タイム/ダッシュ)</th><th>AI指数(100点満点)</th><th>AI印</th><th>Gemini印</th></tr></thead><tbody>"
    
    score_col = f'score_{cond}'
    bonus_col = f'track_bonus_{cond}'

    for i, r in disp_df.iterrows():
        raw_score = r.get(score_col, np.nan)
        is_no_data = r.get('is_no_data_horse', False) or pd.isna(raw_score) or str(raw_score).strip().lower() in ['nan', 'none', '-']
        
        # データ無しの馬は AI印を "-" に固定
        if is_no_data:
            ai_mark = "-"
            b_cls_ai = ""
            ai_mark_str = "<span style='color:#ccc; font-weight:bold;'>-</span>"
        else:
            ai_mark = get_mark(i)
            b_cls_ai = "badge-honmei" if "◎" in ai_mark else "badge-taikou" if "◯" in ai_mark else "badge-tana" if "▲" in ai_mark else "badge-renka" if "△" in ai_mark else "badge-tana" if "☆" in ai_mark else "badge-keshi"
            ai_mark_str = f"<span class='badge-mark {b_cls_ai}'>{ai_mark}</span>"

        gem_mark = r.get('gemini_mark', '-')
        if gem_mark == "-": gem_str = "<span style='color:#ccc; font-weight:bold;'>-</span>"
        else:
            b_cls_gem = "badge-honmei" if "◎" in gem_mark else "badge-taikou" if "◯" in gem_mark else "badge-tana" if "▲" in gem_mark else "badge-renka" if "△" in gem_mark else "badge-tana" if "☆" in gem_mark else "badge-keshi"
            gem_str = f"<span class='badge-mark {b_cls_gem}'>{gem_mark}</span>"

        kyaku = str(r.get('脚質', '-'))
        k_style = "background:#ff7675;" if kyaku == "逃" else "background:#e67e22;" if kyaku == "先" else "background:#3498db;" if kyaku == "差" else "background:#2ecc71;" if kyaku == "追" else "background:#ccc;"

        jockey_name = str(r.get('騎手', r.get('騎手_clean', '-'))).strip()
        j_win = float(r.get('jockey_win_rate', 0.0)) * 100 if str(r.get('jockey_win_rate', 0.0)) != '-' else 0.0
        jockey_str = f"{jockey_name}<br><span style='font-size:0.8em; color:#666;'>({j_win:.1f}%)</span>"
        
        pop_val = r.get('人気', '-')
        odds_val = r.get('オッズ', '-')
        try: pop_str = f"<b>{int(float(pop_val))}</b>人気<br><span style='font-size:0.8em; color:#666;'>({float(odds_val):.1f}倍)</span>"
        except: pop_str = "<span style='color:#ccc;'>-</span>"

        t_idx_val = r.get('ema3_custom_time_index_m', r.get('prev1_time_idx', np.nan))
        s_idx_val = r.get('ema3_custom_start_index', r.get('prev1_start_idx', np.nan))
        
        try:
            t_f, s_f = float(t_idx_val), float(s_idx_val)
            if pd.isna(t_f) or t_f == 0.0: idx_str = "<span class='missing-data'>データ無</span>"
            else: idx_str = f"<span style='font-size:0.85em;'><b>{int(t_f)}</b> / <span style='color:#e67e22;'><b>{int(s_f)}</b></span></span>"
        except: idx_str = "<span class='missing-data'>データ無</span>"

        notes = []
        track_bonus = r.get(bonus_col, 0.0)
        try:
            track_bonus = float(track_bonus)
            if track_bonus > 0: notes.append(f"<span class='track-bias-badge'>☔ 補正 +{track_bonus:.1f}</span>")
            elif track_bonus < 0: notes.append(f"<span class='track-bias-badge' style='background:#e74c3c;'>⚠️ 割引 {track_bonus:.1f}</span>")
        except: pass
        note_str = "<br>".join(notes) if notes else "<span style='color:#ccc;'>-</span>"
        
        u_num_val = r.get('馬番', 0)
        try: u_num = int(float(u_num_val))
        except: u_num = 0
        h_name = str(r.get('馬名', '-')).strip()
        
        if is_no_data:
            score_str = "<span style='color:#ccc;'>-</span>"
        else:
            try: score_str = f"<b>{float(raw_score):.1f}</b>"
            except: score_str = "<span style='color:#ccc;'>-</span>"
        
        rentai_raw = r.get('jockey_rentai_rate', np.nan)
        try: rentai = int(float(rentai_raw) * 100)
        except: rentai = int(j_win * 1.8)

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
<td>{ai_mark_str}</td>
<td>{gem_str}</td>
</tr>"""
    html += "</tbody></table></div>"
    return html

if df_future.empty: 
    st.warning("⚠️ 出馬表データが存在しないか空です。")
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
        st.error("⚠️ キャッシュが見つかりません。")
    else:
        scored_df = pd.DataFrame(cache_data[target_id])
        
        if 'is_no_data_horse' in scored_df.columns:
            no_data_count = int(scored_df['is_no_data_horse'].sum())
        else:
            no_data_count = int(scored_df['ema3_custom_time_index_m'].isna().sum()) if 'ema3_custom_time_index_m' in scored_df.columns else 0
            
        total_horses = len(scored_df)

        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0] if not df_future.empty else {'place_name': '地方', 'r_num': '?'}
        race_display_name = f"{info['place_name']} {info['r_num']}R (指定馬場: {track_condition})"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)

        if no_data_count >= 3:
            st.markdown(f"""
            <div class='warning-banner'>
                ⚠️ 【見送り推奨】このレースは出走 {total_horses} 頭中 {no_data_count} 頭の過去データが存在しません（データ無し馬 3頭以上）。<br>
                予想精度を担保できないため、勝負レースからの除外・見送りを強く推奨します。
            </div>
            """, unsafe_allow_html=True)

        target_score_col = f'score_{track_condition}'
        if target_score_col in scored_df.columns:
            scored_df['sort_key'] = pd.to_numeric(scored_df[target_score_col], errors='coerce')
            scored_df = scored_df.sort_values(by=['sort_key'], ascending=False, na_position='last').drop(columns=['sort_key']).reset_index(drop=True)

        def get_u_num(df, index):
            if len(df) > index:
                val = df.iloc[index].get('馬番', 0)
                try: return int(float(val))
                except: return 0
            return 0

        u_1 = get_u_num(scored_df, 0)
        u_2 = get_u_num(scored_df, 1)
        u_3 = get_u_num(scored_df, 2)
        u_4 = get_u_num(scored_df, 3)
        u_5 = get_u_num(scored_df, 4)

        try:
            s1 = float(scored_df.iloc[0].get(target_score_col, 80))
            s2 = float(scored_df.iloc[1].get(target_score_col, 80))
            score_diff = s1 - s2
        except: score_diff = 0

        if score_diff >= 4:
            rec_pattern_name = "🎯 【絶対能力上位・1着固定流し】 1位 ➔ 2〜4位 (計6点)"
            rec_text = f"1位の強さが抜けている（点数 {score_diff:.1f} 差）ため、頭固定の3連単で狙います。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}"
        else:
            rec_pattern_name = "🛡️ 【能力混戦・1頭軸流し】 1位 ➔ 2〜5位 (計6点)"
            rec_text = f"上位陣が能力拮抗（点数 {score_diff:.1f} 差）しているため、1位軸の3連複で広く狙います。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}, {u_5:02d}"

        st.markdown(f"""
        <div class='rec-banner-formation'>
            {rec_pattern_name}<br>
            <span style='font-size:0.85em; font-weight:normal;'>
            * <b>軸馬(1頭):</b> <b>{axis_horse}</b><br>
            * <b>相手(ヒモ):</b> {target_horses}<br>
            * <b>理由:</b> 🤖 <b>AI指数（{track_condition}馬場補正済）</b>に基づく選定です。{rec_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAI評価 (📍 指定馬場「{track_condition}」補正版)</div>", unsafe_allow_html=True)

        scored_df['gemini_mark'] = "-"
        if target_id in st.session_state['gemini_results']:
            for h_num, g_mark in st.session_state['gemini_results'][target_id].get('marks', {}).items():
                scored_df.loc[pd.to_numeric(scored_df.get('馬番'), errors='coerce') == float(h_num), 'gemini_mark'] = g_mark

        st.empty().markdown(generate_beautiful_table(scored_df, track_condition), unsafe_allow_html=True)

        # Gemini 予想の動的生成機能
        st.markdown("<div class='section-header'>🤖 Gemini AIによる展開・買い目見解分析</div>", unsafe_allow_html=True)
        active_api_key = api_key_input or GEMINI_API_KEY
        
        if not active_api_key:
            st.info("💡 サイドバーに Gemini API Key を入力すると、AI見解の自動生成機能が有効になります。")
        else:
            if st.button("✨ Gemini AIの見解を生成する", use_container_width=True, type="primary"):
                with st.spinner("Geminiが展開と各馬の評価を詳細分析中..."):
                    try:
                        client = genai.Client(api_key=active_api_key)
                        
                        prompt = f"""あなたはプロの地方競馬予想家です。
以下のレース出走データとAI指数（100点満点評価）を分析し、展開予想と推奨買い目をまとめてください。

【レース情報】
{race_display_name}

【出走馬データ (上位順)】
"""
                        for idx, row in scored_df.iterrows():
                            h_no = row.get('馬番', '-')
                            h_nm = row.get('馬名', '-')
                            k_style = row.get('脚質', '-')
                            j_nm = row.get('騎手', '-')
                            sc = row.get(target_score_col, 'データ無')
                            prompt += f"- 馬番{h_no}: {h_nm} | 脚質:{k_style} | 騎手:{j_nm} | AI指数:{sc}\n"

                        prompt += """
【出力フォーマット】
1. 展開予想 (ペース展開、ハナ主張馬、展開向く馬)
2. 最終予想印 (◎, ◯, ▲, △, ☆) とそれぞれの選定理由
3. 推奨買い目 (3連複 / 3連単)
"""
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt
                        )
                        st.session_state['gemini_results'][target_id] = {'text': response.text, 'marks': {}}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gemini API エラー: {e}")

        if target_id in st.session_state['gemini_results']:
            st.markdown(f"<div class='gemini-output-box'>{st.session_state['gemini_results'][target_id]['text']}</div>", unsafe_allow_html=True)