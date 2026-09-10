import os
import re
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

st.set_page_config(page_title="AI予想 勝ち子ちゃん | Ranking(LambdaMART)版", page_icon="🌸", layout="wide")

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
    .gemini-output-box { background-color: #ffffff !important; color: #222222 !important; padding: 20px; border-radius: 12px; border: 2px solid #f2cdd5; margin-top: 15px; }
    .note-icon { font-size: 1.1em; margin-right: 3px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.4, 10])
with col1: st.write("🌸")
with col2: st.title("AI予想 勝ち子ちゃん (Ranking・LambdaMART完全版)")

if 'selected_race_id' not in st.session_state: st.session_state['selected_race_id'] = None
if 'baba_status' not in st.session_state: st.session_state['baba_status'] = "良"
if 'bias_multipliers' not in st.session_state: st.session_state['bias_multipliers'] = {"逃": 1.0, "先": 1.0, "差": 1.0, "追": 1.0}
if 'gemini_results' not in st.session_state: st.session_state['gemini_results'] = {}

def set_race_id(rid): st.session_state['selected_race_id'] = rid
def reset_bias(): st.session_state['bias_multipliers'] = {"逃": 1.0, "先": 1.0, "差": 1.0, "追": 1.0}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

FUTURE_CSV = "future_races_chiho.csv"
MODEL_FILES = ["keiba_ai_model_nar_ensemble.pkl", "keiba_ai_model.pkl"]
NAR_PLACES = {"30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋", "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"}

def clean_horse_name(name_val): 
    if pd.isna(name_val): return ""
    s_val = unicodedata.normalize('NFKC', str(name_val))
    return re.sub(r'[\s・･._\u3000\t\r\n]+', '', s_val).strip()

def parse_weight_info(val):
    if pd.isna(val): return 470.0, 0.0
    s_val = str(val).strip()
    m = re.search(r'(\d{3})(?:\(([-+]?\d+)\))?', s_val)
    if m: return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0
    return 470.0, 0.0

@st.cache_resource
def load_app_data(): 
    model_data = None
    for f in MODEL_FILES:
        if os.path.exists(f):
            model_data = joblib.load(f)
            break
            
    if os.path.exists('past_dicts.pkl'):
        dicts = joblib.load('past_dicts.pkl')
    else:
        st.error("⚠️ エラー: 軽量辞書 (past_dicts.pkl) が見つかりません。")
        dicts = {'jockey_dict':{}, 'horse_dict':{}, 'waku_dict':{}, 'trainer_dict':{}, 'combo_dict':{}, 'place_avg_rank_dict':{}, 'jockey_rentai_dict':{}}

    return model_data, dicts

model_data, loaded_dicts = load_app_data()
jockey_dict = loaded_dicts.get('jockey_dict', {})
horse_dict = loaded_dicts.get('horse_dict', {})
waku_dict = loaded_dicts.get('waku_dict', {})
trainer_dict = loaded_dicts.get('trainer_dict', {})
combo_dict = loaded_dicts.get('combo_dict', {})
place_avg_rank_dict = loaded_dicts.get('place_avg_rank_dict', {})
jockey_rentai_dict = loaded_dicts.get('jockey_rentai_dict', {})

df_future = pd.DataFrame()
if os.path.exists(FUTURE_CSV):
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
        try:
            df_future = pd.read_csv(FUTURE_CSV, low_memory=False, encoding=enc)
            if not df_future.empty: break
        except Exception: continue

if not df_future.empty and 'race_id' in df_future.columns:
    df_future['place_code'] = df_future['race_id'].astype(str).str[4:6]
    df_future['place_name'] = df_future['place_code'].map(NAR_PLACES).fillna("地方")
    df_future['r_num'] = pd.to_numeric(df_future['race_id'].astype(str).str[10:12], errors='coerce').fillna(1).astype(int)
    df_future['day_label'] = df_future['date'].astype(str) if 'date' in df_future.columns else datetime.now().strftime("%Y-%m-%d")

def get_kyakushitsu(fc): 
    return "逃" if fc <= 2.5 else "先" if fc <= 4.5 else "差" if fc <= 7.5 else "追"

def calculate_race_scores(race_id_target, target_df, baba_status="良", bias_dict=None):
    if target_df.empty: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy().reset_index(drop=True)
    if race_df.empty: return None

    race_df['place_code_str'] = race_df['race_id'].astype(str).str[4:6]
    race_df['weight_num'] = pd.to_numeric(race_df.get('斤量'), errors='coerce').fillna(54.0)
    race_df['馬番_num'] = pd.to_numeric(race_df.get('馬番'), errors='coerce').fillna(0)
    race_df['waku_num'] = pd.to_numeric(race_df.get('枠番'), errors='coerce').fillna(0)
    
    race_df['馬名_clean'] = race_df.get('馬名', pd.Series(['']*len(race_df))).astype(str).apply(clean_horse_name)
    race_df['騎手_clean'] = race_df.get('騎手', pd.Series(['']*len(race_df))).astype(str).apply(clean_horse_name)
    race_df['trainer_clean'] = race_df.get('調教師', race_df['騎手_clean']).astype(str).apply(clean_horse_name)
    race_df['jockey_trainer_combo'] = race_df['騎手_clean'] + "_" + race_df['trainer_clean']

    if '馬体重' in race_df.columns:
        parsed_w = race_df['馬体重'].apply(parse_weight_info)
        race_df['body_weight'] = parsed_w.apply(lambda x: x[0])
        race_df['body_weight_diff'] = parsed_w.apply(lambda x: x[1])
    else:
        race_df['body_weight'], race_df['body_weight_diff'] = 470.0, 0.0

    def safe_assign(col_name, dict_key, default_val):
        race_df[col_name] = race_df['馬名_clean'].apply(lambda x: horse_dict.get(x, {}).get(dict_key, default_val))

    safe_assign('horse_prize_avg', 'horse_prize_avg', 0.0)
    race_mean_prize = max(race_df['horse_prize_avg'].mean(), 0.1)
    race_df['race_prize_relative'] = race_df['horse_prize_avg'] / race_mean_prize
    race_df['race_prize_rank'] = race_df['horse_prize_avg'].rank(ascending=False, method='min')
    
    safe_assign('first_corner', 'first_corner', 5.0)
    safe_assign('last_corner', 'last_corner', 5.0)
    race_df['prev_1c'] = race_df['first_corner']
    race_df['race_expected_pace'] = race_df['prev_1c'].mean()

    # 🌟 修復1: 展開分析用の逃げ先行頭数フラグ
    race_df['is_front_runner'] = (race_df['prev_1c'] <= 3.5).astype(int)
    race_df['race_front_runners'] = race_df['is_front_runner'].sum()
    
    safe_assign('prev_time_index_avg', 'prev_time_index_avg', 100.0)
    safe_assign('prev_start_index_avg', 'prev_start_index_avg', 50.0)
    safe_assign('prev_time_sec', 'prev_time_sec', 90.0)
    safe_assign('prev_last3f_sec', 'prev_last3f_sec', 39.0)
    safe_assign('horse_career_runs', 'horse_career_runs', 5.0)
    
    race_df['place_avg_rank'] = race_df.apply(lambda r: place_avg_rank_dict.get((r['馬名_clean'], r['place_code_str']), 7.0), axis=1)
    race_df['jockey_rentai_rate'] = race_df['騎手_clean'].apply(lambda x: jockey_rentai_dict.get(x, 0.2))
    race_df['jockey_win_rate'] = race_df['騎手_clean'].apply(lambda x: jockey_dict.get(x, 0.05))
    race_df['trainer_win_rate'] = race_df['trainer_clean'].apply(lambda x: trainer_dict.get(x, 0.05))

    # 🌟 修復2: UIバッジ用の特注フラグ（鞍上強化・降級戦）
    safe_assign('prev_jockey_win', 'prev_jockey_win', 0.05)
    race_df['jockey_upgrade_diff'] = race_df['jockey_win_rate'] - race_df['prev_jockey_win']
    safe_assign('prev_prize_log', 'prev_prize_log', 0.0)
    race_df['is_class_drop'] = (race_df['prev_prize_log'] - race_mean_prize >= 0.4).astype(int)
    
    race_df['eff_my_time_idx'] = race_df['prev_time_index_avg']
    race_df['weight_change'] = race_df['body_weight_diff']
    race_df['斤量'] = race_df['weight_num']

    diff_cols = ['eff_my_time_idx', 'body_weight', 'trainer_win_rate', 'jockey_win_rate', 'jockey_rentai_rate', 'horse_prize_avg']
    for c in diff_cols:
        mean_v = race_df[c].mean()
        std_v = max(race_df[c].std(ddof=0), 1e-6)
        race_df[f'{c}_race_diff'] = race_df[c] - mean_v
        race_df[f'{c}_race_zscore'] = (race_df[c] - mean_v) / std_v

    race_df['脚質'] = race_df['first_corner'].apply(get_kyakushitsu)
    race_df['jockey_win_display'] = (race_df['jockey_win_rate'] * 100).round(1)

    def safe_predict(model, df):
        if hasattr(model, 'feature_name_'): cols = model.feature_name_
        elif hasattr(model, 'feature_names_in_'): cols = model.feature_names_in_
        elif isinstance(model_data, dict) and 'features' in model_data: cols = model_data['features']
        else: return None
        
        X_pred = pd.DataFrame()
        for c in cols:
            if c in df.columns:
                X_pred[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0.0)
            else:
                X_pred[c] = 0.0
        return model.predict(X_pred.astype(float))

    if not model_data or not isinstance(model_data, dict):
        st.error("⚠️ エラー: AIモデルファイルが正常にロードされていません。")
        st.stop()

    preds = []
    for m_key in ['model_rank_lgb', 'model_rank_xgb', 'model_rank_cat']:
        if model_data.get(m_key):
            preds.append(safe_predict(model_data[m_key], race_df))

    if preds:
        race_df['rank_score_raw'] = np.mean(preds, axis=0)
    else:
        st.error("⚠️ AIモデルから予測値を出力できませんでした。")
        st.stop()

    score_mean = race_df['rank_score_raw'].mean()
    score_std = max(race_df['rank_score_raw'].std(ddof=0), 1e-6)
    race_df['score_disp'] = np.round(((race_df['rank_score_raw'] - score_mean) / score_std) * 10 + 50).astype(int)

    return race_df.sort_values(by=['rank_score_raw'], ascending=False).reset_index(drop=True)
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
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>特注</th><th>騎手(勝率)</th><th>脚質</th><th>連対率</th><th>指数実績<br>(タイム/ダッシュ)</th><th>AI偏差値</th><th>AI印</th><th>Gemini印</th></tr></thead><tbody>"
    
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

        jockey_str = f"{r.get('騎手', '-')}<br><span style='font-size:0.8em; color:#666;'>({r.get('jockey_win_display', 0.0)}%)</span>"
        
        t_idx = int(r.get('prev_time_index_avg', 100))
        s_idx = int(r.get('prev_start_index_avg', 50))
        idx_str = f"<span style='font-size:0.85em;'><b>{t_idx}</b> / <span style='color:#e67e22;'><b>{s_idx}</b></span></span>"

        # 🌟 前半で復活補修した特注サインとバッジ表示
        notes = []
        if r.get('eff_my_time_idx_race_diff', 0) >= 5.0: notes.append("<span class='note-icon'>⚡</span>時計上位")
        if r.get('track_dist_ema_index', 100.0) >= 105.0: notes.append("<span class='note-icon'>📍</span>コース巧者")
        if r.get('jockey_upgrade_diff', 0.0) >= 0.08: notes.append("<span class='note-icon'>🚀</span>勝負気配(鞍上)")
        if r.get('trainer_win_rate_race_diff', 0.0) >= 0.1: notes.append("<span class='note-icon'>🏢</span>勝負気配(厩舎)")
        if r.get('is_class_drop', 0) == 1: notes.append("<span class='note-icon'>💰</span>降級戦")
        
        note_str = "<br>".join(notes) if notes else "<span style='color:#ccc;'>-</span>"

        html += f"""<tr>
<td style='font-weight:bold; color:#c94a65 !important;'>{int(r['馬番_num']):02d}</td>
<td style='text-align:left; font-weight:800; color:#5a3d46 !important;'>{r.get('馬名', '-')}</td>
<td style='font-size:0.85em; font-weight:bold; color:#e67e22;'>{note_str}</td>
<td style='color:#666666 !important;'>{jockey_str}</td>
<td><span style='{k_style} color:#fff !important; padding:3px 8px; border-radius:6px; font-size:0.85em; font-weight:bold;'>{kyaku}</span></td>
<td style='color:#5a3d46 !important;'><b>{r.get('horse_rentai_display', 0.0)}%</b></td>
<td>{idx_str}</td>
<td style='color:#5a3d46 !important; font-size:1.1em;'><b>{int(r['score_disp'])}</b></td>
<td><span class='badge-mark {b_cls_ai}'>{ai_mark}</span></td>
<td>{gem_str}</td>
</tr>"""
    html += "</tbody></table></div>"
    return html

if df_future.empty: st.warning("⚠️ 出馬表データ (future_races_chiho.csv) が存在しないか空です。")
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
                            st.success("✅ 馬場バイアスを認識しました！（※PythonのAI偏差値は歪めず、Geminiの独立予想にのみ活用します）")
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
        <br><span style='font-size:0.85em; color:#666;'>※この倍率はGeminiの展開読みにのみ使用され、AIの純粋な能力値(偏差値)は歪めません。</span>
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

        score_diff = scored_df.iloc[0]['score_disp'] - scored_df.iloc[1]['score_disp'] if len(scored_df) > 1 else 0
        
        front_runners_count = int(scored_df.iloc[0]['race_front_runners']) if 'race_front_runners' in scored_df.columns else 0
        pace_text = f"<br>🔥 <b>展開予想:</b> このレースは逃げ・先行馬が {front_runners_count} 頭います。{'ハイペース崩れに注意！差し馬の評価を上げています。' if front_runners_count >= 3 else 'ペースは落ち着きそうです。前残り注意。'}"

        if score_diff >= 4:
            rec_pattern_name = "🎯 【絶対能力上位・1着固定流し】 1位 ➔ 2〜4位 (計6点)"
            rec_text = f"1位の強さが抜けている（偏差値 {score_diff} 差）ため、迷わず頭固定の3連単で仕留めます。"
            axis_horse = f"{u_1:02d}"
            target_horses = f"{u_2:02d}, {u_3:02d}, {u_4:02d}"
        else:
            rec_pattern_name = "🛡️ 【能力混戦・1頭軸流し】 1位 ➔ 2〜5位 (計6点)"
            rec_text = f"上位陣が能力拮抗（偏差値 {score_diff} 差）しているため、1位を軸にしつつ相手を広く構えた3連複で狙います。"
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
        target_id_str = str(target_id)
        if target_id_str in st.session_state['gemini_results']:
            saved_marks = st.session_state['gemini_results'][target_id_str].get('marks', {})
            for h_num, g_mark in saved_marks.items():
                scored_df.loc[scored_df['馬番_num'] == h_num, 'gemini_mark'] = g_mark

        table_placeholder = st.empty()
        table_placeholder.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)

        if st.button("🎀 Gemini独自の予想（＋α展開・騎手・格 重視）を生成", use_container_width=True):
            if not api_key_input: 
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            table_summary = []
            for idx, row in scored_df.head(9).iterrows():
                table_summary.append(
                    f"【Python評価 {idx+1}位】 馬番:{int(row['馬番_num']):02d} | 馬名:{row['馬名']} | 脚質:{row['脚質']} | 騎手:{row['騎手']} | 鞍上強化スコア:{row.get('jockey_upgrade_diff', 0.0):.2f} | 降級フラグ:{int(row.get('is_class_drop', 0))} | 前走タイム偏差値:{row.get('eff_my_time_idx_race_diff', 0.0):.1f} | 騎手連対率:{row.get('jockey_rentai_rate', 0.0):.2f} | Python偏差値:{row['score_disp']}"
                )

            sys_inst = f"""あなたは地方競馬の熟練予想AI「勝ち子ちゃん（Gemini）」です。
Python（機械学習AI）がオッズを見ずに「過去の絶対能力（偏差値）」だけで弾き出した上位9頭のデータをお渡しします。

【🚨あなたの役割と絶対厳守のルール🚨】
あなたの役割は、Pythonと同じ視点で予想することではありません。
Pythonのスコアをベースにしつつも、あなたは【＋αの激アツ要素（前走タイム偏差値、鞍上強化、降級、馬場バイアス）】を加味して、全く別の角度から独自の印（◎, ◯, ▲, △, ☆）を打ってください。

1. Pythonの評価順（1位〜5位）と全く同じ順序で印を打つことは禁止します。別視点の予想家として独立した評価を下してください。
2. データ内の「前走タイム偏差値（プラスが大きいほど優秀）」「鞍上強化スコア（高いほど勝負気配）」「降級フラグ(1)」を持つ馬は高く評価してください。
3. 今の馬場バイアスを加味し、展開が向く実力馬・伏兵（☆）を1頭見つけ出してください。

競馬場: {info['place_name']} / 馬場: {st.session_state['baba_status']}
適用中のバイアス: 逃げ {bm.get('逃')}倍, 先行 {bm.get('先')}倍, 差し {bm.get('差')}倍, 追込 {bm.get('追')}倍

【回答の構成】
🌸 Geminiの独自見解（特注サイン・展開フォーカス）
（Pythonの数値だけでは測れない「前走タイム」や「鞍上の勝負気配」から、どうレースを読むかを簡潔に）

🎯 Gemini独自の印と解説
※【重要】システムが馬番を自動抽出するため、必ず以下のフォーマット通りに記述してください。馬番は必ず半角数字にし、[ ]で囲んでください。
◎ [馬番] 馬名 （地力・前走レベル・コース適性・展開から最も狙える理由）
◯ [馬番] 馬名 （理由）
▲ [馬番] 馬名 （理由）
△ [馬番] 馬名 （理由）
☆ [馬番] 馬名 （Python評価は低めだが、展開や勝負気配で一発ある理由）
"""
            with st.spinner("🎀 Geminiが『前走タイム・騎手・降級』に基づく独自予想を作成中..."):
                try:
                    ai_client = genai.Client(api_key=api_key_input)
                    response = ai_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"対象レース: {race_display_name}\n\n対象馬:\n" + "\n".join(table_summary),
                        config=types.GenerateContentConfig(system_instruction=sys_inst, temperature=0.6) 
                    )
                    resp_text = response.text
                    
                    new_marks = {}
                    for line in resp_text.split('\n'):
                        match = re.search(r'([◎◯▲△☆]).*?\[(\d{1,2})\]', line)
                        if match:
                            mark = match.group(1)
                            horse_num = int(match.group(2))
                            new_marks[horse_num] = mark
                            
                    st.session_state['gemini_results'][target_id_str] = {
                        'marks': new_marks,
                        'text': resp_text
                    }
                    
                    for h_num, g_mark in new_marks.items():
                        scored_df.loc[scored_df['馬番_num'] == h_num, 'gemini_mark'] = g_mark
                    table_placeholder.markdown(generate_beautiful_table(scored_df), unsafe_allow_html=True)
                    
                except Exception as e: st.error(f"エラー: {e}")

        if target_id_str in st.session_state['gemini_results']:
            clean_text = re.sub(r'^[#\-\s]+', '', st.session_state['gemini_results'][target_id_str]['text'].strip())
            st.markdown(f"<div class='gemini-output-box'>{clean_text}</div>", unsafe_allow_html=True)
            