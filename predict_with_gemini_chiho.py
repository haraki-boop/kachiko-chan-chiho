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
st.set_page_config(page_title="AI予想 勝ち子ちゃん | キャッシュ表示版", page_icon="🌸", layout="wide")

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
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (APIキャッシュ超高速版)")

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
        return joblib.load(CACHE_FILE)
    else:
        st.error(f"⚠️ エラー: キャッシュファイル '{CACHE_FILE}' が見つかりません。")
        return {}

df_future = load_future_data()
cache_data = load_cache()

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
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>特注</th><th>騎手(勝率)</th><th>脚質</th><th>人気(オッズ)</th><th>連対率</th><th>指数実績<br>(タイム/ダッシュ)</th><th>AI偏差値</th><th>AI印</th><th>Gemini印</th></tr></thead><tbody>"
    
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

        t_idx_val = r.get('eff_my_time_idx', r.get('prev_my_time_idx', 100))
        s_idx_val = r.get('eff_my_start_idx', r.get('custom_start_index', 50))
        
        is_missing = False
        try:
            t_f, s_f = float(t_idx_val), float(s_idx_val)
            if pd.isna(t_f) or pd.isna(s_f) or (t_f == 40.0 and s_f == 50.0) or (t_f == 0.0 and s_f == 0.0) or (t_f == 100.0 and s_f == 50.0):
                is_missing = True
        except:
            is_missing = True

        if is_missing:
            idx_str = "<span class='missing-data'>データ無</span>"
            t_idx = 40
        else:
            t_idx = int(t_f)
            s_idx = int(s_f)
            idx_str = f"<span style='font-size:0.85em;'><b>{t_idx}</b> / <span style='color:#e67e22;'><b>{s_idx}</b></span></span>"

        notes = []
        if not is_missing and t_idx >= 105: notes.append("<span class='note-icon'>⚡</span>時計上位")
        note_str = "<br>".join(notes) if notes else "<span style='color:#ccc;'>-</span>"
        
        u_num_val = r.get('馬番_num', r.get('馬番', r.get('gate_num', 0)))
        u_num = int(float(u_num_val)) if pd.notna(u_num_val) else 0
        h_name = str(r.get('馬名', r.get('馬名_clean', '-'))).strip()
        score = int(float(r.get('score_disp', 50))) if pd.notna(r.get('score_disp', 50)) else 50
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
<td style='color:#5a3d46 !important; font-size:1.1em;'><b>{score}</b></td>
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
        st.error(f"⚠️ このレース（{target_id}）の予測キャッシュが見つかりません。")
    else:
        scored_df = pd.DataFrame(cache_data[target_id])
        
        if 'score_disp' in scored_df.columns:
            scored_df['score_disp'] = pd.to_numeric(scored_df['score_disp'], errors='coerce').fillna(50)
            scored_df = scored_df.sort_values(by='score_disp', ascending=False).reset_index(drop=True)
        
        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0] if not df_future.empty else {'place_name': '地方', 'r_num': '?', 'race_name': ''}
        race_display_name = f"{info['place_name']} {info['r_num']}R 【{info.get('race_name', '')}】"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        missing_count = 0
        for _, row_data in scored_df.iterrows():
            try:
                t = float(row_data.get('eff_my_time_idx', row_data.get('prev_my_time_idx', 40)))
                s = float(row_data.get('eff_my_start_idx', row_data.get('custom_start_index', 50)))
                if (t == 40.0 and s == 50.0) or pd.isna(t) or t == 0.0 or (t == 100.0 and s == 50.0):
                    missing_count += 1
            except:
                missing_count += 1
        
        if missing_count >= 3:
            st.warning(f"⚠️ **【見送り推奨】** このレースは過去データがない（初出走・転入など）、またはデータが正常に取得できていない馬が **{missing_count}頭** 含まれています。AI予想のブレが大きくなるため、勝負を避けることを強くおすすめします。")

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

        if len(scored_df) > 1:
            s1 = float(scored_df.iloc[0].get('score_disp', 50))
            s2 = float(scored_df.iloc[1].get('score_disp', 50))
            score_diff = s1 - s2
        else:
            score_diff = 0

        if score_diff >= 4:
            rec_pattern_name = "🎯 【絶対能力上位・1着固定流し】 1位 ➔ 2〜4位 (計6点)"
            rec_text = f"1位の強さが抜けている（偏差値 {score_diff:.1f} 差）ため、迷わず頭固定の3連単で仕留めます。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}"
        else:
            rec_pattern_name = "🛡️ 【能力混戦・1頭軸流し】 1位 ➔ 2〜5位 (計6点)"
            rec_text = f"上位陣が能力拮抗（偏差値 {score_diff:.1f} 差）しているため、1位を軸にしつつ相手を広く構えた3連複で狙います。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}, {u_5:02d}"

        st.markdown(f"""
        <div class='rec-banner-formation'>
            {rec_pattern_name}<br>
            <span style='font-size:0.85em; font-weight:normal;'>
            * <b>軸馬(1頭):</b> <b>{axis_horse}</b><br>
            * <b>相手(ヒモ):</b> {target_horses}<br>
            * <b>理由:</b> 🤖 <b>LambdaMARTが算出した純粋な絶対強さ(偏差値)</b>に基づく選定です。{rec_text}{pace_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='section-header'>📊 勝ち子ちゃんのAI評価 (📍 絶対能力・偏差値版)</div>", unsafe_allow_html=True)

        scored_df['gemini_mark'] = "-"
        if target_id in st.session_state['gemini_results']:
            saved_marks = st.session_state['gemini_results'][target_id].get('marks', {})
            for h_num, g_mark in saved_marks.items():
                scored_df.loc[pd.to_numeric(scored_df.get('馬番_num', scored_df.get('馬番', scored_df.get('gate_num'))), errors='coerce') == float(h_num), 'gemini_mark'] = g_mark

        table_placeholder = st.empty()
        table_placeholder.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Gemini独自の完全独立予想（世論・血統・直感重視）を生成", use_container_width=True):
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
                bbs = str(row.get('世論コメント', row.get('bbs_comment', '特になし')))
                
                # 💡 ゴミコメント（空欄のデフォルトテキスト）を弾くフィルターを追加
                if "コメントを投稿する" in bbs or "取得不可" in bbs:
                    bbs = "特になし"
                
                # 💡 AI偏差値も一緒に渡すように追加
                ai_score = row.get('score_disp', 50)
                
                table_summary.append(
                    f"馬番:{u_n:02d} | 馬名:{row.get('馬名', row.get('馬名_clean', ''))} | 脚質:{row.get('脚質', '')} | 騎手:{row.get('騎手', row.get('騎手_clean', ''))} | AI偏差値:{ai_score} | 人気/オッズ:{pop}人気({odds}倍) | ネットの評価:{bbs}"
                )

            race_distance = info.get('distance', '不明')
            
            # 💡 完全に無視するのではなく「AI偏差値」をベースにしつつ、展開やオッズで予想させるプロンプトに変更
            sys_inst = f"""あなたは地方競馬の事情通であり、AIのデータと競馬のセオリーを融合させて最終結論を出す天才予想家「勝ち子ちゃん（Gemini）」です。
出走馬の基本データ、システムの算出した「AI偏差値」、そして「オッズ・世論」をお渡しします。

【🚨あなたの役割と絶対厳守のルール🚨】
今回のあなたの予想では、システムの「AI偏差値（絶対能力）」をベースとして尊重しつつ、そこに「展開（脚質の偏り）」「オッズの歪み（妙味）」「騎手やコース適性」を掛け合わせて、最終的な印を打ってください。
完全にAIを無視するのではなく、「AI評価が高いのに人気がないから美味しい」「AI評価は高いが、逃げ馬多数で展開が厳しそうだから対抗に下げる」といった、データとリアルを融合させた現実的なアプローチをしてください。
※ネットの評価が「特になし」の場合は、無理に世論について言及する必要はありません。

【回答の構成】
🌸 Geminiの独自見解（ペース予想、AI偏差値とオッズのギャップ、狙い目など）
🎯 Gemini独自の印と解説
※【重要】システムが馬番を自動抽出するため、必ず以下のフォーマット通りに記述してください。馬番は必ず半角数字にし、[ ]で囲んでください。
◎ [馬番] 馬名 （理由）
◯ [馬番] 馬名 （理由）
▲ [馬番] 馬名 （理由）
△ [馬番] 馬名 （理由）
☆ [馬番] 馬名 （理由）
"""
            with st.spinner("🎀 Geminiがリアルタイムの世論とオッズを加味して思考中..."):
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
            clean_text = re.sub(r'^[#\-\s]+', '', st.session_state['gemini_results'][target_id]['text'].strip())
            st.markdown(f"<div class='gemini-output-box'>{clean_text}</div>", unsafe_allow_html=True)