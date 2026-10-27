import os
import re
import json
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from google import genai
from google.genai import types

# ==========================================
# 定数・設定
# ==========================================
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
FUTURE_CSV: str = "future_races_chiho.csv"
CACHE_FILE: str = "app_cache_chiho.pkl"
NAR_PLACES: Dict[str, str] = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

st.set_page_config(page_title="地方競馬AI 勝ち子ちゃん | コース物理バイアス完全対応", page_icon="🏇", layout="wide")

# ==========================================
# 競馬場・距離別 物理力学詳細データベース
# ==========================================
COURSE_PHYSICS_DB: Dict[Tuple[str, int], Dict[str, Any]] = {
    # --- 大井競馬場 ---
    ("大井", 1600): {
        "summary": "【大井1600m・内枠死に枠】内回り使用。スタート後100mで1角。1〜2枠は外から殺到されて包囲・砂被り確定。7〜8枠の好位差し・外先行が跳ねる特異コース。",
        "bias": {1: -2.0, 2: -1.5, 3: -0.5, 4: 0.0, 5: 0.5, 6: 1.0, 7: 1.5, 8: 2.0},
        "target_style": "外枠好位差し・外先行"
    },
    ("大井", 1200): {
        "summary": "【大井1200m・外枠先行有利】直線約386m。スタート後に緩いカーブがあり、外からスムーズに加速した外枠先行馬が被せて押し切る。内枠は揉まれやすい。",
        "bias": {1: -0.5, 2: -0.5, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.5, 8: 1.5},
        "target_style": "外枠先行・好位差し"
    },
    ("大井", 1400): {
        "summary": "【大井1400m・フラット〜外目スムーズ】外回り。枠の極端な有利不利は少ないが、直線が長いため外からスムーズに伸びる先行・差しが優勢。",
        "bias": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0},
        "target_style": "先行・好位差し"
    },
    ("大井", 1800): {
        "summary": "【大井1800m・外回りスタミナ戦】1角までの距離が長く枠差は少ない。ペースが落ち着きやすく、スタミナのある先行・好位差しが安定。",
        "bias": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 0.0, 6: 0.5, 7: 0.5, 8: 0.5},
        "target_style": "先行・好位"
    },

    # --- 浦和競馬場 ---
    ("浦和", 1400): {
        "summary": "【浦和1400m・内枠先行絶対】直線220m。4角先頭〜2番手が黄金ルート。1〜3枠の逃げ先行が絶対有利。7〜8枠は1角で大きく外に振られて距離ロス甚大。差しは前崩壊時でも3着まで。",
        "bias": {1: 2.5, 2: 2.0, 3: 1.5, 4: 0.5, 5: -0.5, 6: -1.0, 7: -1.5, 8: -2.0},
        "target_style": "内枠逃げ・先行"
    },
    ("浦和", 1500): {
        "summary": "【浦和1500m・内枠有利】4コーナー奥ポケットからの発走。3コーナーまでの距離が長めだが、小回りのため内枠（1〜3枠）先行が極めて有利。",
        "bias": {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.5},
        "target_style": "内枠先行"
    },
    ("浦和", 800): {
        "summary": "【浦和800m・電撃戦】テンのダッシュ力と内枠が全て。1〜2枠でハナを取った馬がそのまま押し切る。",
        "bias": {1: 2.5, 2: 2.0, 3: 1.0, 4: 0.0, 5: -0.5, 6: -1.0, 7: -1.5, 8: -2.0},
        "target_style": "逃げ"
    },

    # --- 川崎競馬場 ---
    ("川崎", 900): {
        "summary": "【川崎900m・最内枠逃げ絶対】2コーナー手前ポケット発走後すぐにコーナー。1〜2枠のハナ主張馬が絶対的有利。外枠は外を回らされてノーチャンス。",
        "bias": {1: 2.5, 2: 2.0, 3: 1.0, 4: 0.0, 5: -0.5, 6: -1.0, 7: -1.5, 8: -2.0},
        "target_style": "内枠逃げ"
    },
    ("川崎", 1400): {
        "summary": "【川崎1400m・先行有利】4号スタンド前発走。1角までの距離が長く枠の差は小さいが、タイトなコーナーのため先行有利。",
        "bias": {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
        "target_style": "逃げ・先行"
    },
    ("川崎", 1500): {
        "summary": "【川崎1500m・好位差し台頭】コーナー6回。先行争いが激化しやすく、1〜2角で無理をした逃げ馬が潰れる。好位〜中団の外差しが突き抜ける波乱コース。",
        "bias": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0},
        "target_style": "好位差し・外差し"
    },
    ("川崎", 1600): {
        "summary": "【川崎1600m・1角ポケット発走】スタート直後にカーブを迎えるため、内枠（1〜3枠）がロスなく回れる。ただしペースが上がると前潰れも。",
        "bias": {1: 1.5, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: 0.0, 7: -0.5, 8: -0.5},
        "target_style": "内枠好位"
    },

    # --- 船橋競馬場 ---
    ("船橋", 1000): {
        "summary": "【船橋1000m・内枠ダッシュ型】ワンターン戦。内ラチ沿いをロスなく直進できる1〜3枠の先行が圧倒的に有利。",
        "bias": {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.5},
        "target_style": "内枠逃げ"
    },
    ("船橋", 1200): {
        "summary": "【船橋1200m・スパイラルカーブ】3〜4角のスパイラルカーブによりコーナーで減速しにくい。先行馬が残るが、外から差す馬も十分届く。",
        "bias": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5},
        "target_style": "先行・好位差し"
    },
    ("船橋", 1600): {
        "summary": "【船橋1600m・外枠差し届く】直線の攻防が長く、外枠からスムーズに勢いをつけた差し馬が3〜4角でポジションを押し上げて台頭する。",
        "bias": {1: -0.5, 2: -0.5, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.5},
        "target_style": "外枠好位差し"
    },

    # --- 高知競馬場 ---
    ("高知", 1400): {
        "summary": "【高知1400m・イン砂地獄・外枠大外一気】内ラチ沿いの砂が意図的に深く盛られており、内を通ると失速。1〜2枠は致命的。外枠（6〜8枠）から外目をスムーズに先行・捲り差す馬が圧勝する特異コース。",
        "bias": {1: -2.5, 2: -1.5, 3: -0.5, 4: 0.0, 5: 0.5, 6: 1.5, 7: 2.0, 8: 2.5},
        "target_style": "外枠先行・外捲り差し"
    },
    ("高知", 1300): {
        "summary": "【高知1300m・外枠有利】1400m同様にインの砂が重い。内枠で砂を被った人気馬は惨敗リスク大。外目先行〜外差しが鉄則。",
        "bias": {1: -2.0, 2: -1.5, 3: -0.5, 4: 0.0, 5: 0.5, 6: 1.5, 7: 2.0, 8: 2.0},
        "target_style": "外枠先行・外差し"
    },
    ("高知", 1600): {
        "summary": "【高知1600m・外差しスタミナ戦】タフな馬場での長丁場。インを避けて大外を回して上がってくる捲り差し馬の独壇場。",
        "bias": {1: -2.0, 2: -1.0, 3: 0.0, 4: 0.5, 5: 1.0, 6: 1.5, 7: 2.0, 8: 2.0},
        "target_style": "外捲り差し"
    },

    # --- 園田・姫路競馬場 ---
    ("園田", 1400): {
        "summary": "【園田1400m・内枠先行絶対】一周1051m、直線213m。小回りタイト。内枠（1〜3枠）先行がそのまま雪崩れ込む。差し馬は3角手前から大捲りを打たないと間に合わない。",
        "bias": {1: 2.0, 2: 1.5, 3: 1.5, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.5},
        "target_style": "内枠逃げ・先行"
    },
    ("園田", 1230): {
        "summary": "【園田1230m・内枠ダッシュ戦】2角ポケット発走。内枠先行がそのまま最短距離を回って粘り切る。",
        "bias": {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.5},
        "target_style": "内枠先行"
    },
    ("園田", 1700): {
        "summary": "【園田1700m・先行好位】ホームストレッチ発走で1角まで十分。スタミナのある先行・好位差しが安定。",
        "bias": {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
        "target_style": "先行・好位"
    },

    # --- 新名古屋競馬場（弥富） ---
    ("名古屋", 1500): {
        "summary": "【新名古屋1500m・1枠砂重・外枠好位有利】弥富移転で直線が240mへ延長。最内1枠の砂が重く、外枠（6〜8枠）から外目をスムーズに先行・好位差しする馬が圧倒的に有利。",
        "bias": {1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.5, 8: 1.5},
        "target_style": "外枠好位差し・先行"
    },
    ("名古屋", 920): {
        "summary": "【新名古屋920m・超電撃スプリント】内枠〜中枠のスタートダッシュが速い逃げ馬が有利。",
        "bias": {1: 1.5, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.0},
        "target_style": "内枠逃げ"
    },

    # --- 佐賀競馬場 ---
    ("佐賀", 1400): {
        "summary": "【佐賀1400m・最内砂地獄・外枠先行有利】最内1頭分の砂が深く重い。1枠の人気馬がハナを叩けないと砂被りで即沈没。外枠（6〜8枠）から揉まれず運べる先行・好位差しが黄金ルート。",
        "bias": {1: -2.0, 2: -1.0, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.5, 7: 2.0, 8: 2.0},
        "target_style": "外枠先行・外好位差し"
    },
    ("佐賀", 1300): {
        "summary": "【佐賀1300m・外枠先行有利】1400m同様、最内の砂が深い。外枠からスムーズに先手を奪える馬が有利。",
        "bias": {1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.5, 8: 1.5},
        "target_style": "外枠先行"
    },

    # --- 笠松競馬場 ---
    ("笠松", 1400): {
        "summary": "【笠松1400m・内枠先行有利】直線約201m（地方最短級）。1角ポケット発走。内枠（1〜3枠）先行がロスなく立ち回って押し切る。",
        "bias": {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.5},
        "target_style": "内枠逃げ・先行"
    },

    # --- 金沢競馬場 ---
    ("金沢", 1400): {
        "summary": "【金沢1400m・イン重・外枠好位】ラチ沿いの砂が重く、最内1枠はマイナス。外枠先行〜外好位差しが走りやすい。",
        "bias": {1: -1.5, 2: -0.5, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.5, 8: 1.5},
        "target_style": "外枠先行・好位"
    },

    # --- 門別競馬場 ---
    ("門別", 1200): {
        "summary": "【門別1200m・外枠先行・外差し】直線330mと外回り広大。外枠からスピードに乗せて外目を追走できる馬が有利。",
        "bias": {1: -0.5, 2: 0.0, 3: 0.0, 4: 0.5, 5: 0.5, 6: 1.0, 7: 1.0, 8: 1.0},
        "target_style": "外枠先行・好位差し"
    },
    ("門別", 1800): {
        "summary": "【門別1800m・実力スタミナ戦】大外回りで直線も長い。コース自体の枠有利不利は小さく、スタミナと持続力が直結。",
        "bias": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
        "target_style": "先行・好位差し"
    },

    # --- 盛岡・水沢競馬場 ---
    ("盛岡", 1200): {
        "summary": "【盛岡1200mダート・ワンターン広大】ワンターンで直線約300m。コース幅が広く枠差は小さいが、先行集団が残りやすい。",
        "bias": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
        "target_style": "逃げ・先行"
    },
    ("水沢", 1400): {
        "summary": "【水沢1400m・小回り内枠先行】小回り平坦コース。内枠先行がそのまま押し切るケースが多い。",
        "bias": {1: 1.5, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0, 6: -0.5, 7: -1.0, 8: -1.0},
        "target_style": "内枠先行"
    }
}

# ==========================================
# CSSスタイリング
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #fafbfc !important; color: #333333 !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    h1 { font-size: 1.8rem !important; color: #2c3e50 !important; font-weight: 800; }
    h2 { font-size: 1.3rem !important; color: #34495e !important; }
    .section-header { font-size: 1.2rem; font-weight: 800; color: #2c3e50 !important; margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }
    
    .table-container { width: 100%; overflow-x: auto; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); background-color: #ffffff; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; font-size: 0.9em; }
    .kachi-table thead tr { background: #2c3e50; color: #ffffff !important; }
    .kachi-table th { padding: 10px 8px; text-align: center; border-right: 1px solid rgba(255,255,255,0.1); color: #ffffff !important; font-size: 0.85em; font-weight: 600; }
    .kachi-table td { padding: 8px 8px; text-align: center; border-bottom: 1px solid #edf2f7; color: #2d3748 !important; }
    .kachi-table tbody tr:hover td { background: #f8fafc; }
    
    .badge-mark { color: #fff !important; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.82em; display: inline-block; min-width: 50px; text-align: center; }
    .badge-honmei { background: #e74c3c; }
    .badge-taikou { background: #3498db; }
    .badge-tana   { background: #2ecc71; }
    .badge-renka  { background: #f39c12; }
    .badge-ana    { background: #9b59b6; }
    .badge-danger { background: #d63031; color: #fff !important; font-weight: 900; }
    .badge-pass   { background: #b2bec3; color: #fff !important; }
    .badge-keshi  { background: #e2e8f0; color: #718096 !important; }
    
    .course-banner { background: #edf2f7; border-left: 4px solid #3182ce; padding: 12px 16px; border-radius: 6px; margin-bottom: 15px; font-size: 0.92rem; color: #2b6cb0; font-weight: bold; }
    .upset-banner { background: linear-gradient(135deg, #d63031, #e17055); color: #ffffff !important; padding: 16px 20px; border-radius: 8px; font-size: 1.15rem; font-weight: bold; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(214, 48, 49, 0.25); }
    .solid-banner { background: linear-gradient(135deg, #636e72, #b2bec3); color: #ffffff !important; padding: 14px 20px; border-radius: 8px; font-size: 1.05rem; font-weight: bold; margin-bottom: 20px; }
    .ticket-badge { background: #fff3cd; color: #856404; padding: 3px 8px; border-radius: 4px; font-weight: bold; border: 1px solid #ffeeba; }
    .weight-trigger { font-size: 0.85em; font-weight: 600; color: #2d3436; background: #f1f2f6; padding: 4px 6px; border-radius: 4px; text-align: left; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# セッション初期化
# ==========================================
if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None
if 'gemini_eval_cache' not in st.session_state:
    st.session_state['gemini_eval_cache'] = {}

def set_race_id(rid: str) -> None:
    st.session_state['selected_race_id'] = str(rid)

# ==========================================
# データロード関数
# ==========================================
@st.cache_data
def load_future_data() -> pd.DataFrame:
    if not os.path.exists(FUTURE_CSV):
        return pd.DataFrame()
    encodings = ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']
    for enc in encodings:
        try:
            df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, low_memory=False, encoding=enc)
            if not df.empty:
                df['race_id'] = pd.to_numeric(df['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)
                df['place_code'] = df['race_id'].astype(str).str[4:6]
                df['place_name'] = df['place_code'].map(NAR_PLACES).fillna("地方")
                df['r_num'] = pd.to_numeric(df['race_id'].str[10:12], errors='coerce').fillna(1).astype(int)
                df['day_label'] = df['date'].astype(str).str.strip() if 'date' in df.columns else "当日"
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_resource
def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try: return joblib.load(CACHE_FILE)
        except Exception: pass
    return {}

def clean_name(text: Any) -> str:
    if pd.isna(text): return ""
    import unicodedata
    s = unicodedata.normalize('NFKC', str(text))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

# ==========================================
# コース物理バイアス計算（超拡張版）
# ==========================================
def get_course_physics(place_name: str, distance_num: Any) -> Dict[str, Any]:
    p = str(place_name).strip()
    try: d = int(float(distance_num))
    except (ValueError, TypeError): d = 1400

    # 1. 完全一致検索
    if (p, d) in COURSE_PHYSICS_DB:
        return COURSE_PHYSICS_DB[(p, d)]
    
    # 2. 近似距離フォールバック
    for (cp, cd), data in COURSE_PHYSICS_DB.items():
        if cp == p and abs(cd - d) <= 100:
            return data
            
    # 3. 競馬場デフォルトフォールバック
    default_bias = {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}
    return {
        "summary": f"【{p}{d}m】地方競馬標準コース。小回り構造のため先行優位が基本。",
        "bias": default_bias,
        "target_style": "逃げ・先行"
    }

def calculate_frame_bias_enhanced(place_name: str, distance_num: Any, wakuban: Any, mode: str) -> Tuple[float, str]:
    try: w = int(float(wakuban))
    except (ValueError, TypeError): return 0.0, ""

    if mode == "フラット (補正なし)": return 0.0, ""
    elif mode == "内枠超有利 (+2.0 / -1.0)": return (2.0 if w in [1, 2] else 1.0 if w in [3, 4] else -1.0 if w in [7, 8] else 0.0), "内枠優遇"
    elif mode == "外枠超有利 (-1.0 / +2.0)": return (-1.0 if w in [1, 2] else 1.0 if w in [5, 6] else 2.0 if w in [7, 8] else 0.0), "外枠優遇"

    c_info = get_course_physics(place_name, distance_num)
    bias_map = c_info.get("bias", {})
    bonus = bias_map.get(w, 0.0)
    
    reason = "枠有利" if bonus >= 1.0 else "枠不利" if bonus <= -1.0 else "枠並"
    return round(float(bonus), 1), reason

# ==========================================
# ユーティリティ・バッジ関数
# ==========================================
def get_py_mark_badge(idx: int) -> str:
    marks = {0: ("◎ 本命", "badge-honmei"), 1: ("◯ 対抗", "badge-taikou"), 
             2: ("▲ 単穴", "badge-tana"), 3: ("△ 押さえ", "badge-renka")}
    mark, css_class = marks.get(idx, ("消", "badge-keshi"))
    return f"<span class='badge-mark {css_class}'>{mark}</span>"

def get_gemini_verdict_badge(role: str) -> str:
    r = str(role).strip()
    if "危険" in r or "消" in r: return f"<span class='badge-mark badge-danger'>⚠️ {r}</span>"
    if "逆転" in r or "◎" in r or "本命" in r: return f"<span class='badge-mark badge-honmei'>🎯 {r}</span>"
    if "穴" in r or "☆" in r: return f"<span class='badge-mark badge-ana'>☆ {r}</span>"
    if "相手" in r or "◯" in r or "連下" in r: return f"<span class='badge-mark badge-taikou'>{r}</span>"
    if "見送り" in r or "静観" in r: return f"<span class='badge-mark badge-pass'>{r}</span>"
    return f"<span class='badge-mark badge-keshi'>{r if r else '-'}</span>"

def get_frame_badge_html(wakuban: Any) -> str:
    try: w = int(float(wakuban))
    except (ValueError, TypeError): return "<span style='color:#ccc;'>-</span>"
    colors = {
        1: ("#ffffff", "#000000", "1px solid #bbb"), 2: ("#222222", "#ffffff", "none"),
        3: ("#e74c3c", "#ffffff", "none"), 4: ("#3498db", "#ffffff", "none"),
        5: ("#f1c40f", "#222222", "none"), 6: ("#2ecc71", "#ffffff", "none"),
        7: ("#e67e22", "#ffffff", "none"), 8: ("#fd79a8", "#ffffff", "none"),
    }
    bg, fg, border = colors.get(w, ("#eee", "#333", "none"))
    return f"<span style='background:{bg}; color:{fg}; border:{border}; padding:2px 7px; border-radius:4px; font-weight:bold;'>{w}</span>"

# ==========================================
# 対比テーブル HTML 生成
# ==========================================
def generate_comparison_table(disp_df: pd.DataFrame) -> str:
    html = ["<div class='table-container'><table class='kachi-table'>"]
    html.append("<thead><tr><th>枠</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>人気(オッズ)</th><th>Python評価<br>(AI指数/印)</th><th style='background:#1b2a47;'>Gemini物理判定</th><th style='background:#1b2a47; text-align:left; min-width:180px;'>⚖️ 直前馬体重トリガー</th><th style='background:#1b2a47; text-align:left; min-width:320px;'>📝 診断根拠 (展開・枠・近5走)</th></tr></thead><tbody>")
    
    for i, r in disp_df.iterrows():
        raw_score = r.get('final_score', np.nan)
        is_no_data = pd.isna(raw_score) or str(raw_score).strip().lower() in ['nan', 'none', '-']
        
        # Python評価
        score_str = "-" if is_no_data else f"{float(raw_score):.1f}"
        py_mark = get_py_mark_badge(i) if not is_no_data else "<span style='color:#ccc;'>-</span>"
        py_col_str = f"<b>{score_str}</b> pt<br>{py_mark}"
        
        # Gemini物理判定
        gem_role = r.get('gemini_role', '-')
        gem_badge = get_gemini_verdict_badge(gem_role)
        gem_weight = r.get('gemini_weight_trigger', '-')
        gem_reason = r.get('gemini_reason', '-')
        
        weight_html = f"<div class='weight-trigger'>{gem_weight}</div>" if gem_weight != '-' else "<span style='color:#ccc;'>-</span>"
        reason_html = f"<div style='font-size:0.88em; line-height:1.4;'>{gem_reason}</div>" if gem_reason != '-' else "<span style='color:#ccc;'>-</span>"
        
        w_badge = get_frame_badge_html(r.get('枠番'))
        u_num = int(float(r.get('馬番', 0))) if pd.notna(r.get('馬番')) else 0
        h_name = str(r.get('馬名', '-')).strip()
        kyaku = str(r.get('公式脚質', r.get('脚質', '-')))
        
        try: pop_str = f"<b>{int(float(r.get('人気', 0)))}</b>人気<br><span style='font-size:0.8em; color:#666;'>({float(r.get('オッズ', 0)):.1f}倍)</span>"
        except: pop_str = "<span style='color:#ccc;'>-</span>"

        html.append(f"<tr><td>{w_badge}</td><td style='font-weight:bold; color:#2c3e50;'>{u_num:02d}</td><td style='text-align:left; font-weight:800;'>{h_name}</td><td><b>{kyaku}</b></td><td>{pop_str}</td><td>{py_col_str}</td><td>{gem_badge}</td><td style='text-align:left;'>{weight_html}</td><td style='text-align:left;'>{reason_html}</td></tr>")

    html.append("</tbody></table></div>")
    return "".join(html)

# ==========================================
# Gemini独立分析 (コース物理辞書 + 生データ注入)
# ==========================================
def run_gemini_physics_analysis(api_key: str, df: pd.DataFrame, target_id: str, race_info: str, track_condition: str, course_physics: Dict[str, Any]) -> None:
    client = genai.Client(api_key=api_key)
    
    horse_raw_summaries = []
    for _, r in df.iterrows():
        past_texts = []
        for i in range(1, 6):
            if f"前走{i}_着順" in r and pd.notna(r[f"前走{i}_着順"]):
                p_rank = r.get(f"前走{i}_着順", "")
                p_baba = r.get(f"前走{i}_馬場", "")
                p_time = r.get(f"前走{i}_タイム", "")
                p_pass = r.get(f"前走{i}_通過順", "")
                p_up3f = r.get(f"前走{i}_上がり3F", "")
                p_wt = r.get(f"前走{i}_馬体重", "")
                p_diff = r.get(f"前走{i}_体重増減", "")
                if p_rank or p_time:
                    past_texts.append(f"前走{i}:{p_rank}着(馬場:{p_baba}/タイム:{p_time}/通過:{p_pass}/上3F:{p_up3f}/体重:{p_wt}kg({p_diff}))")
        
        p_history = " / ".join(past_texts) if past_texts else "近走データなし"
        
        h_info = (
            f"枠:{r.get('枠番','-')} | 馬番:{int(float(r.get('馬番',0))):02d} | 馬名:{r.get('馬名','')} | "
            f"脚質:{r.get('公式脚質', r.get('脚質','不明'))} | 斤量:{r.get('斤量','')} | 騎手:{r.get('騎手','')} | "
            f"人気:{r.get('人気','-')}人気(オッズ:{r.get('オッズ','-')}倍) | "
            f"【近5走時系列】{p_history}"
        )
        horse_raw_summaries.append(h_info)

    system_instruction = f"""
あなたは地方競馬のコース物理力学・砂質バイアス・隊列分析に特化した冷徹なプロトラックマンです。
AIスコア等の事前計算値には一切頼らず、提供された「枠番」「公式脚質」「人気」「近5走の通過順・時計・馬場・体重推移」と、以下の【コース物理特性】のみを用いて、上位人気馬が物理的・構造的に飛ぶ（着外に沈む）爆弾レースを特定してください。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当レースのコース物理特性・砂質バイアス】
{course_physics.get('summary', '地方標準小回り')}
推奨脚質・有利ゾーン: {course_physics.get('target_style', '先行')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【厳格な評価アルゴリズム】
1. コース死角×枠番判定：
   - 上記の【コース物理特性】と各馬の枠番を厳格に照合すること。
   - 例: 大井1600mの内枠先行、高知の最内1〜2枠、佐賀1枠などは、砂被り・包囲で即死する「危険消し」の最有力候補。
   - 例: 浦和1400mの外枠先行は、外回しロスでスタミナ切れの危険馬。
2. 隊列・ハナ争い判定（前潰れの有無）：
   - 通過順履歴からハナを主張する馬をカウント。逃げ・先行が3頭以上同居していれば激突前潰れ確定。
3. 差し馬の抽出（3着ヒモ穴・逆転穴）：
   - 前走大敗でも「上がり3F最速実績」や「通過順の押し上げ（捲り）」がある差し馬は、前崩壊時のヒモ穴【☆穴抜擢】に固定。
4. 直前馬体重トリガーの算出：
   - 近5走の好走時・凡走時の体重レンジから、「当日何kg以上なら買い / 何kg以下なら消し」を全頭に対して具体的に算出すること。

【判定ロール（役割）の割り当てルール】
・「危険消し」：人気先行馬で、コース死角枠・同型激突・馬場不適・体重消耗のいずれかで飛ぶリスクが極めて高い馬。
・「逆転軸◎」：コース有利枠を引き、前潰れを好位外目で追走、または重馬場最速時計を持つ逆転候補。
・「☆穴抜擢」：展開崩壊時に後方から雪崩れ込む伏兵（上がり最速実績馬など）。
・「相手◯/△」：展開不問で走れる相手候補。
・「静観/消し」：上記以外の馬。

※出力は必ず以下のJSON形式のみとし、Markdownの```json等は絶対に含めないでください。
{{
  "is_upset_race": true, // 上位人気が飛ぶ波乱レースかどうか (true/false)
  "scenario_summary": "（展開・隊列・コースバイアスによる有利不利の物理的予測を2行以内で要約）",
  "strict_ticket": "（例: ワイド 06 - 09 (1点) / 3連複 06 - 02, 09 (2点) / 危険馬切りのため購入見送り など）",
  "horses_evaluation": [
    {{
      "馬名": "馬名",
      "gemini_role": "危険消し", // 危険消し / 逆転軸◎ / ☆穴抜擢 / 相手◯ / 静観
      "weight_trigger": "前走465kg(-8kg)で大敗。460kg以下なら疲弊決定で完全消し。472kg以上戻していれば再考",
      "gemini_reason": "最内枠で外に逃げ馬2頭。コース特性上ハナを叩かれて砂を被る構造で前潰れ必至。"
    }}
  ]
}}
"""

    prompt = f"""
【分析対象レース】
レース情報: {race_info}
当日の想定馬場: 【 {track_condition} 】

【出走全馬の物理生データ】
{chr(10).join(horse_raw_summaries)}
"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3
        )
    )
    
    res_text = response.text if response.text else ""
    res_text = re.sub(r'^```(?:json)?\n', '', res_text)
    res_text = re.sub(r'\n```$', '', res_text).strip()
    match = re.search(r'\{.*\}', res_text, re.DOTALL)
    if match: res_text = match.group(0)
    
    gemini_data = json.loads(res_text)
    
    eval_map = {}
    for h in gemini_data.get("horses_evaluation", []):
        h_clean = clean_name(h.get("馬名", ""))
        eval_map[h_clean] = {
            "role": h.get("gemini_role", "-"),
            "weight_trigger": h.get("weight_trigger", "-"),
            "reason": h.get("gemini_reason", "-")
        }
        
    st.session_state['gemini_eval_cache'][target_id] = {
        "is_upset": gemini_data.get("is_upset_race", False),
        "scenario": gemini_data.get("scenario_summary", ""),
        "ticket": gemini_data.get("strict_ticket", ""),
        "horses": eval_map
    }

# ==========================================
# メイン処理
# ==========================================
df_future = load_future_data()
cache_data = load_cache()

st.sidebar.header("⚙️ 設定・コンディション")
api_key_input = st.sidebar.text_input("Gemini API Key", value=GEMINI_API_KEY, type="password")
track_condition = st.sidebar.radio("当日の想定馬場", options=["良", "稍重", "重", "不良"], index=0)

st.sidebar.markdown("---")
frame_bias_mode = st.sidebar.selectbox("枠順バイアス補正モード", options=["自動 (コース物理DB連動)", "内枠超有利 (+2.0 / -1.0)", "外枠超有利 (-1.0 / +2.0)", "フラット (補正なし)"], index=0)

if st.sidebar.button("🔄 キャッシュクリア", use_container_width=True): 
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if df_future.empty:
    st.warning("⚠️ 出馬表データが見つかりません。")
else:
    st.markdown("<div class='section-header'>🎯 予想レース選択</div>", unsafe_allow_html=True)
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
                        f"{r}R", key=f"btn_{rid}", use_container_width=True, 
                        type="primary" if st.session_state['selected_race_id'] == rid else "secondary",
                        on_click=set_race_id, args=(rid,)
                    )

target_id = st.session_state['selected_race_id']
if target_id:
    st.markdown("---")
    if target_id not in cache_data:
        st.error("⚠️ 指定レースのAI推論キャッシュ(app_cache_chiho.pkl)が見つかりません。")
    else:
        scored_df = pd.DataFrame(cache_data[target_id])
        target_score_col = f'score_{track_condition}'
        
        info = df_future[df_future['race_id'].astype(str) == target_id].iloc[0] if not df_future.empty else {'place_name': '地方', 'r_num': '?', 'distance': 1400}
        place_name_val = info.get('place_name', '地方')
        dist_num_val = info.get('distance', 1400)

        # 競馬場・距離別 物理力学データの取得
        course_physics = get_course_physics(place_name_val, dist_num_val)

        # Python スコア再計算（コース物理DB連動補正）
        frame_bonuses, final_scores = [], []
        for _, row in scored_df.iterrows():
            f_bonus, _ = calculate_frame_bias_enhanced(place_name_val, dist_num_val, row.get('枠番'), frame_bias_mode)
            base_s = pd.to_numeric(row.get(target_score_col, row.get('score_disp_base', 70)), errors='coerce')
            frame_bonuses.append(f_bonus)
            final_scores.append(min(100.0, round(base_s + f_bonus, 1)) if pd.notna(base_s) else np.nan)

        scored_df['frame_bonus'] = frame_bonuses
        scored_df['final_score'] = final_scores
        scored_df = scored_df.sort_values(by='final_score', ascending=False, na_position='last').reset_index(drop=True)
        
        # 近5走データのマージ
        if not df_future.empty:
            race_future = df_future[df_future['race_id'] == target_id].copy()
            race_future['馬名_clean'] = race_future['馬名'].apply(clean_name)
            scored_df['馬名_clean'] = scored_df['馬名'].apply(clean_name)
            
            merge_cols = ['馬名_clean'] + [c for c in race_future.columns if '前走' in c or '公式脚質' in c or '人気' in c or 'オッズ' in c]
            merge_cols = list(dict.fromkeys(merge_cols))
            scored_df = pd.merge(scored_df, race_future[merge_cols], on='馬名_clean', how='left')

        race_display_name = f"{place_name_val} {info.get('r_num','?')}R ダート{dist_num_val}m (指定馬場: {track_condition})"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        # コース物理特性バナー表示
        st.markdown(f"<div class='course-banner'>🧭 {course_physics.get('summary')}</div>", unsafe_allow_html=True)

        # Gemini 診断結果の結合
        g_eval = st.session_state['gemini_eval_cache'].get(target_id)
        if g_eval:
            if g_eval.get("is_upset"):
                st.markdown(f"""
                <div class='upset-banner'>
                    💥 【爆弾レース警告】上位人気の崩壊シナリオを検知！<br>
                    <span style='font-size:0.9em; font-weight:normal;'>{g_eval.get('scenario')}</span><br>
                    <div style='margin-top:8px;'><b>🎟 Gemini厳選買い目:</b> <span class='ticket-badge'>{g_eval.get('ticket')}</span></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='solid-banner'>
                    🛡 【順当・波乱低め】先行馬によるマイペース決着濃厚（見送り推奨）<br>
                    <span style='font-size:0.9em; font-weight:normal;'>{g_eval.get('scenario')}</span><br>
                    <div style='margin-top:6px;'><b>🎟 判断:</b> <span class='ticket-badge'>{g_eval.get('ticket')}</span></div>
                </div>
                """, unsafe_allow_html=True)

            horses_data = g_eval.get("horses", {})
            for idx, r in scored_df.iterrows():
                h_c = clean_name(r.get('馬名'))
                if h_c in horses_data:
                    scored_df.at[idx, 'gemini_role'] = horses_data[h_c].get('role', '-')
                    scored_df.at[idx, 'gemini_weight_trigger'] = horses_data[h_c].get('weight_trigger', '-')
                    scored_df.at[idx, 'gemini_reason'] = horses_data[h_c].get('reason', '-')

        # 対比テーブル描画
        st.markdown("<div class='section-header'>⚔️ Python (確率・能力値) × Gemini (物理・馬体重診断) 激突対比表</div>", unsafe_allow_html=True)
        st.markdown(generate_comparison_table(scored_df), unsafe_allow_html=True)

        # Gemini 解析実行ボタン
        active_api_key = api_key_input or GEMINI_API_KEY
        if active_api_key:
            if st.button("🔥 Geminiに生データを渡し、上位人気の崩壊判定＆馬体重トリガーを診断", use_container_width=True, type="primary"):
                with st.spinner(f"Geminiが【{place_name_val}{dist_num_val}m】のコース物理力学・隊列・馬体重バイオリズムを完全独立分析中..."):
                    try:
                        run_gemini_physics_analysis(active_api_key, scored_df, target_id, race_display_name, track_condition, course_physics)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gemini解析エラー: {e}")
        else:
            st.info("💡 サイドバーに Gemini API Key を入力すると、Geminiの物理判定・馬体重診断機能が利用可能になります。")