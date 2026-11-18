import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import unicodedata
from datetime import datetime
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple

# ==========================================
# 🎨 アプリの基本設定 & スタイル定義
# ==========================================
st.set_page_config(page_title="地方競馬AI 勝ち子ちゃん (馬連特化・道悪/物理タグ優遇版)", page_icon="🏇", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%); color: #333333; }
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #2c3e50; margin: 1.2rem 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }
    .table-container { overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; font-size: 15px; }
    .kachi-table th { padding: 10px 8px; text-align: center; background: #f8f9fa; border-bottom: 2px solid #dee2e6; font-size: 15px; font-weight: bold; color: #333; }
    .kachi-table td { padding: 8px 6px; text-align: center; border-bottom: 1px solid #f1f3f5; font-size: 15px; color: #333; vertical-align: middle; }
    
    .badge-mark { color: #fff; padding: 4px 8px; border-radius: 6px; font-weight: bold; display: inline-block; min-width: 45px; font-size: 14px; text-align: center; }
    .badge-honmei { background: #e74c3c; } .badge-taikou { background: #3498db; }
    .badge-tana { background: #2ecc71; } .badge-renka { background: #f39c12; }
    .badge-ana { background: #9b59b6; } .badge-chuu { background: #e67e22; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
    
    .waku-badge { display: inline-block; width: 24px; height: 24px; line-height: 24px; border-radius: 4px; text-align: center; font-weight: bold; font-size: 14px; }
    .multiplier-up { color: #e74c3c; font-weight: bold; }
    .multiplier-down { color: #3498db; font-weight: bold; }
    .multiplier-flat { color: #7f8c8d; }

    .sense-card { background-color: #ffffff; border-left: 6px solid #8e44ad; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-card-ken { background-color: #fff5f5; border-left: 6px solid #e53e3e; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-card-hot { background-color: #fffff0; border-left: 6px solid #d69e2e; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .sense-title { font-size: 1.15rem; font-weight: bold; color: #8e44ad; margin-bottom: 8px; }
    
    .gemini-view-card { background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 16px 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .gemini-view-title { color: #2980b9; font-size: 1.1rem; font-weight: bold; margin-bottom: 8px; }
    .gemini-view-text { font-size: 15px; color: #333; line-height: 1.6; margin: 0; }
    .course-banner { background: #edf2f7; border-left: 4px solid #3182ce; padding: 10px 14px; border-radius: 6px; margin-bottom: 15px; font-size: 0.95rem; color: #2b6cb0; font-weight: bold; }

    .shinba-badge { background-color: #e67e22; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }
    .age2-badge { background-color: #16a085; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }
    .old-badge { background-color: #2980b9; color: white; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-weight: bold; margin-left: 8px; vertical-align: middle; }

    .strategy-badge-main { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: white; padding: 4px 10px; border-radius: 5px; font-size: 13px; font-weight: bold; display: inline-block; margin-right: 6px; }
    .strategy-badge-sub { background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%); color: white; padding: 4px 10px; border-radius: 5px; font-size: 13px; font-weight: bold; display: inline-block; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔑 APIキー取得（サイドバー入力 + secrets + 環境変数）
# ==========================================
env_key = os.environ.get("GEMINI_API_KEY", "")
secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
default_key = env_key or secret_key

input_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    value=default_key,
    type="password",
    help="Google AI Studioで取得したAPIキーを入力してください"
)
GEMINI_API_KEY = input_key.strip()

FUTURE_CSV = "future_races_chiho.csv"
CACHE_FILE = "app_cache_chiho.pkl"
HISTORY_CSV = "prediction_history_chiho.csv"

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

# ==========================================
# 📊 全14場 コース物理特性DB（トラックバイアス網羅版）
# ==========================================
COURSE_PHYSICS_DB: Dict[Tuple[str, int], Dict[str, Any]] = {
    # 南関東4場
    ("大井", 1200): {"fav_waku": [6, 7, 8], "fav_style": ["逃", "先"], "summary": "【大井1200m・外枠先行有利】直線約386m。外枠先行馬が被せて押し切る。内枠の差し追込は砂を被り厳しい。"},
    ("大井", 1400): {"fav_waku": [5, 6, 7, 8], "fav_style": ["逃", "先", "差"], "summary": "【大井1400m・外スムーズ先行差し】内枠で包まれるとロス大。中〜外枠からスムーズに運べる先行・好位差し優勢。"},
    ("大井", 1600): {"fav_waku": [6, 7, 8], "fav_style": ["先", "差"], "summary": "【大井1600m・内回り砂被り注意】1〜2枠は砂被りリスク甚大。7〜8枠の好位差し・外先行が特注。"},
    ("大井", 1800): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【大井1800m・内枠ロス軽減】1角までの距離があり内枠先行馬が経済コースを立ち回れる。"},
    ("川崎", 900):  {"fav_waku": [1, 2, 3], "fav_style": ["逃"], "summary": "【川崎900m・内枠電撃逃げ】スタートから1角までが短く、1〜3枠の快速逃げ馬が圧倒的有利。"},
    ("川崎", 1400): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【川崎1400m・イン先行絶対】タイトなスパイラルカーブ。1〜3枠の逃げ・先行馬が最短距離で粘り込む。"},
    ("川崎", 1500): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【川崎1500m・内枠先行有利】外枠は1角で外を回される距離ロスが響く。内枠先行が基本。"},
    ("川崎", 1600): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【川崎1600m・内枠先行絶対】内ラチ沿いを確保できる1〜3枠の先行馬が圧倒的有利。"},
    ("船橋", 1200): {"fav_waku": [6, 7, 8], "fav_style": ["逃", "先"], "summary": "【船橋1200m・外枠先行有利】スパイラルカーブでスピードが落ちにくく、外枠から被せていく先行馬が好走。"},
    ("船橋", 1600): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先", "差"], "summary": "【船橋1600m・内〜中枠立ち回り】1角までの距離が長く枠順差は小さめだが、内〜中枠の先行・好位差しが安定。"},
    ("浦和", 800):  {"fav_waku": [1, 2, 3], "fav_style": ["逃"], "summary": "【浦和800m・内枠逃げ一択】1〜3枠のスタート速い逃げ馬がそのまま押し切る。"},
    ("浦和", 1400): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【浦和1400m・内枠先行絶対】1〜3枠の逃げ先行が絶対有利。7〜8枠は3〜4角で外を回され距離ロス甚大。"},
    ("浦和", 1500): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【浦和1500m・内枠先行絶対】スタート直後にコーナーがあり、1〜3枠の先行馬が圧倒的有利。"},
    # 北海道・東北・北陸
    ("門別", 1000): {"fav_waku": [6, 7, 8], "fav_style": ["逃", "先"], "summary": "【門別1000m・外枠スピード有利】砂が深く、外から被せてスピードに乗れる6〜8枠先行が有利。"},
    ("門別", 1200): {"fav_waku": [6, 7, 8], "fav_style": ["先", "差"], "summary": "【門別1200m・外差しも効くパワー馬場】直線の攻防が長く、外枠からスムーズに加速できる先行・差しが優勢。"},
    ("門別", 1700): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【門別1700m・内枠先行粘り込み】1角までの距離が短く、内枠先行馬がペースを握りやすい。"},
    ("盛岡", 1200): {"fav_waku": [6, 7, 8], "fav_style": ["逃", "先"], "summary": "【盛岡1200m・外枠先行有利】ワンターンでスピード勝負。外枠から砂を被らず先行する馬が優勢。"},
    ("盛岡", 1400): {"fav_waku": [5, 6, 7, 8], "fav_style": ["先", "差"], "summary": "【盛岡1400m・起伏あるタフコース】高低差が大きく直線も長い。外から伸びる好位差しが台頭。"},
    ("盛岡", 1600): {"fav_waku": [5, 6, 7, 8], "fav_style": ["先", "差"], "summary": "【盛岡1600m・ワンターン外枠有利】内枠は包まれやすく、外枠の先行・差し馬が能力を発揮しやすい。"},
    ("水沢", 1300): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【水沢1300m・超小回り内枠先行】直線が短く小回り。1〜3枠の逃げ・先行馬がそのまま粘り込む。"},
    ("水沢", 1400): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【水沢1400m・イン先行絶対】1〜3枠の先行馬が圧倒的有利。外枠の差し・追込は届きにくい。"},
    ("水沢", 1600): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【水沢1600m・内枠先行有利】4角ポケット発走で1角までが短く、内枠先行馬が圧倒的有利。"},
    ("金沢", 1400): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【金沢1400m・イン前残り顕著】砂が軽くスピードが出やすい。1〜4枠の逃げ・先行馬が押し切る。"},
    ("金沢", 1500): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【金沢1500m・前残りコース】3〜4角で前が止まりにくく、内〜中枠の先行馬が圧倒的有利。"},
    # 東海・兵庫・四国・九州
    ("笠松", 1400): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【笠松1400m・超小回りインベタ絶対】3〜4角のインをロスなく回る1〜3枠の逃げ・先行馬が圧倒的有利。"},
    ("笠松", 1600): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【笠松1600m・内枠先行有利】小回りで外を回すと距離ロス大。1〜3枠の先行粘り込みが基本。"},
    ("名古屋", 920):  {"fav_waku": [1, 2, 3], "fav_style": ["逃"], "summary": "【名古屋920m・内枠電撃戦】1〜3枠のテンの速い逃げ馬がそのまま押し切る。"},
    ("名古屋", 1500): {"fav_waku": [5, 6, 7, 8], "fav_style": ["逃", "先"], "summary": "【名古屋1500m・外枠先行有利】コース改修後、直線が長く外枠からスムーズに先行できる馬が好走。"},
    ("名古屋", 1700): {"fav_waku": [5, 6, 7, 8], "fav_style": ["逃", "先", "差"], "summary": "【名古屋1700m・外枠スムーズ立ち回り】砂が深くスタミナ要。外枠から被されずに運べる先行・好位差し優勢。"},
    ("園田", 820):  {"fav_waku": [1, 2, 3], "fav_style": ["逃"], "summary": "【園田820m・内枠逃げ絶対】1〜3枠の快速逃げ馬がハナを切ればまず止まらない。"},
    ("園田", 1230): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【園田1230m・内枠先行有利】3角ポケット発走で1〜4枠の先行馬が最短距離で粘り込む。"},
    ("園田", 1400): {"fav_waku": [1, 2, 3], "fav_style": ["逃", "先"], "summary": "【園田1400m・内枠先行絶対】小回りタイト。内枠(1〜3枠)先行がそのまま雪崩れ込む。"},
    ("姫路", 1400): {"fav_waku": [1, 2, 3, 4], "fav_style": ["逃", "先"], "summary": "【姫路1400m・内枠先行有利】園田同様の小回り平坦。1〜4枠の先行馬がそのまま押し切る。"},
    ("高知", 1300): {"fav_waku": [6, 7, 8], "fav_style": ["逃", "先", "差"], "summary": "【高知1300m・イン砂地獄】内ラチ沿いの砂が深く失速しやすい。6〜8枠から外を通る先行・捲り差しが圧勝。"},
    ("高知", 1400): {"fav_waku": [6, 7, 8], "fav_style": ["逃", "先", "差"], "summary": "【高知1400m・イン砂地獄】内枠(1〜3枠)は深い砂で失速。外枠(6〜8枠)からの先行・捲り差しが圧勝。"},
    ("高知", 1600): {"fav_waku": [6, 7, 8], "fav_style": ["先", "差"], "summary": "【高知1600m・外枠捲り差し有利】内を避けて外から長く脚を使える6〜8枠が圧倒的有利。"},
    ("佐賀", 1300): {"fav_waku": [5, 6, 7, 8], "fav_style": ["逃", "先"], "summary": "【佐賀1300m・内砂深い外先行有利】インコースの砂が深く避けるのがセオリー。5〜8枠の先行馬が有利。"},
    ("佐賀", 1400): {"fav_waku": [5, 6, 7, 8], "fav_style": ["逃", "先", "差"], "summary": "【佐賀1400m・外枠有利・イン避け】1〜2枠は深い砂を走らされる。中〜外枠から外目を通す先行・差しが優勢。"}
}

def get_course_physics_info(place_name: str, distance_num: Any) -> Dict[str, Any]:
    p = str(place_name).strip()
    try: d = int(float(distance_num))
    except: d = 1400
    if (p, d) in COURSE_PHYSICS_DB:
        return COURSE_PHYSICS_DB[(p, d)]
    for (kp, kd), val in COURSE_PHYSICS_DB.items():
        if kp == p and abs(kd - d) <= 200:
            return {"fav_waku": val["fav_waku"], "fav_style": val["fav_style"], "summary": f"【{p}{d}m・準拠バイアス】{val['summary']}"}
    return {"fav_waku": [], "fav_style": ["逃", "先"], "summary": f"【{p}{d}m】地方標準ダートコース。基本的に先行力と砂を被らない位置取りが重要。"}

# ==========================================
# 📊 Pydanticによる係数出力スキーマ
# ==========================================
class HorseCoefficient(BaseModel):
    馬名: str = Field(description="評価対象の馬名（データと完全一致）")
    補正係数: float = Field(
        ge=0.5, le=1.6,
        description="定性根拠に基づく確率補正倍率。通常レースは0.50〜1.60（確定タグなしは1.30上限）。新馬/初出走戦は絶対評価値0.50〜1.60。"
    )
    短評: str = Field(
        description="なぜその補正倍率を付与したかの物理的・定性的理由（コース枠バイアス、隊列位置、近5走の生ログ・時計・上がり・馬場適性評価、斤量・馬体重増減など）"
    )

class RaceCoefficientResult(BaseModel):
    summary: str = Field(
        description="レース全体の物理バイアス（先行争いの激化度、内外砂の有利不利、クラス特性）と重要補正馬の統括見解（300文字以内）"
    )
    coefficients: List[HorseCoefficient] = Field(
        description="出走全頭分の補正係数リスト（全頭必須・欠落厳禁）"
    )

# ==========================================
# 🧠 システムプロンプト（近5走生ログ＋道悪適性＋物理タグ優遇版）
# ==========================================
base_system_instruction = """
# 役割定義
あなたは統計モデル（Python）が算出したベース勝率に対し、直前の物理・定性データ（コース別枠順バイアス、隊列力学、近5走の生ログ、道悪適性、斤量・馬体重増減）から乗算補正倍率を付与する「地方競馬精密レバレッジ鑑定AI」です。
印を決める必要はありません。通常レースではレース全体の平均倍率が【1.00前後】に収束するようメリハリをつけ、各馬の好走確率を何倍にすべきか【0.50 〜 1.60】の補正倍率を客観的根拠のみで精密に鑑定してください。

# 厳格な共通鑑定ルール
1. 【インフレ防止（メリハリの徹底）】: 通常レースにおいて全頭に1.1以上のプラス倍率を乱発してはならない。加点馬と同程度の減点馬（0.75〜0.90）を作り、強弱を明確にせよ。
2. 【特大レバレッジ（>1.30）の厳格制限】: 1.30倍を超える高倍率は、データ内に「🔥【コース枠ドンピシャ】」「🔥【道悪巧者】」「🔥【有利枠×着順詐欺】」「🔥【着順詐欺判定】」「【減量恩恵】」の確定物理タグを持つ馬、または明確な敗因・馬場替わりからの条件好転馬にのみ厳格に限定せよ。単なる先行馬というだけでは1.30を上限とせよ。
3. 【逆バイアス・砂被りリスクの減点】: 提供された「コース物理バイアス」に逆行する馬（例：高知・佐賀・大井1600で1〜2枠に入った揉まれ弱い馬、浦和・園田・笠松・水沢で7〜8枠に入った先行馬）には必ず0.70〜0.88のマイナス倍率を付与せよ。
4. 【先行激化（ハナ争い共倒れ）の減点】: 逃げ・先行馬が4頭以上揃いハイペース必至の場合、同型同士の潰し合いリスクを考慮して逃げ馬を0.80〜0.90に減点し、好位差し馬を加点せよ。
5. 【近5走生ログ・道悪適性・馬体重増減の精密読解】:
   - 想定馬場が「重・不良」の場合、近3走以内に重・不良で好走（1〜3着または好時計）している道悪巧者を積極的に高評価（1.32〜1.50）せよ。
   - コース有利枠に入り、かつ前走・前々走で不利や先行潰れにより着順を落とした「有利枠×着順詐欺」馬は、一変の特注穴馬として1.35〜1.55を付与せよ。
   - 極端な馬体重増減（±12kg以上の太め残り・大幅馬体減）は体調リスクとして減点考慮せよ。
6. 【オッズや人気の完全無視】: オッズや人気には一切左右されず、物理事実（枠番・通過順・上がり3F・走破時計・馬場実績・斤量・馬体重）のみを信じよ。
7. 【出力形式と短評】: 出走全頭分を欠落なく出力し、短評には近走戦歴の具体根拠（前走通過順・上がり・馬場適性・枠順）を簡潔に明記せよ。
"""

shinba_addon = """
# 【特注モード：新馬戦・初出走戦（過去走データなし）】
- Python勝率は無効化されているため無視し、枠順のコース物理バイアス、騎手・厩舎実績、斤量条件のみから各馬の好走評価値（0.50〜1.60）を単独鑑定せよ。
- コースの有利枠（浦和・園田・笠松の内枠、高知・大井1200の外枠など）に入った馬に1.35〜1.55を与え、不利枠の馬は0.65〜0.85として明確に差をつけよ。
"""

age2_addon = """
# 【特注モード：2歳戦・3歳若馬戦 — 成長力・コース枠バイアス重視】
若馬戦はキャリアが浅く指数が未成熟なため、「コース物理バイアスへの合致」と「近走の上がり・時計の良化度」が結果を左右する。以下の基準で配分せよ：
- 1.40 〜 1.60（一変・物理ドンピシャ）:
  「🔥【コース枠ドンピシャ】」「🔥【道悪巧者】」「🔥【有利枠×着順詐欺】」を持ち、前走より時計・位置取りが良化している成長馬。
- 1.15 〜 1.30（標準プラス補正）:
  コースの有利枠に入り、安定して先行（1角3番手以内）できる馬（※確定タグなしの上限は1.30まで）。
- 0.95 〜 1.05（中立）: 平均的な内容の馬。
- 0.70 〜 0.90（マイナス補正）: コースの不利枠に入った馬、または前走大敗で時計が全く足りていない馬。
"""

old_horse_addon = """
# 【特注モード：古馬戦（C3〜A級・重賞）— コースバイアス・着順詐欺・道悪適性重視】
古馬戦はすでに基礎能力が固まっており、「コース物理バイアスとの完全一致」「道悪適性」「近5走生ログの不利・着順詐欺」が最も馬券に直結する。以下の基準で配分せよ：
- 1.45 〜 1.60（特注・極限レバレッジ / 1レース最大0〜2頭まで）:
  「🔥【コース枠ドンピシャ】」「🔥【道悪巧者】」「🔥【有利枠×着順詐欺】」のいずれかと「【減量恩恵】」や好時計が噛み合った特注馬。
- 1.32 〜 1.44（大幅プラス補正）:
  確定タグ（「🔥【コース枠ドンピシャ】」「🔥【道悪巧者】」「🔥【有利枠×着順詐欺】」「🔥【着順詐欺判定】」「【減量恩恵】」）を持ち、条件好転・馬場替わりが顕著な馬。
- 1.10 〜 1.30（標準プラス補正）:
  枠順・脚質・馬場実績がコース特性に合致しており、スムーズな先行・好位差しが見込める馬（※確定タグなしの上限は1.30まで）。
- 0.95 〜 1.05（中立）: Pythonベース勝率通りに評価すべき平均的な馬。出走馬の約半数はここに収めよ。
- 0.70 〜 0.90（マイナス補正）:
  コースの不利枠に入った馬、同型過多でハナ争いの共倒れリスクが高い馬、大幅馬体重増減、または酷量（58kg以上）で条件悪化した馬。
"""

# ==========================================
# 🔧 前処理・物理タグ・近5走ログ整形ヘルパー
# ==========================================
def clean_name(text: Any) -> str:
    if pd.isna(text): return ""
    return re.sub(r'[\s・･.\-ー_]+', '', unicodedata.normalize('NFKC', str(text))).strip().upper()

def format_body_weight_clean(raw_bw: Any) -> str:
    """'458(-6)11.63' のような文字列から '458(-6)' のみを綺麗に抽出"""
    if pd.isna(raw_bw): return "-"
    s = str(raw_bw).strip()
    m = re.match(r'^(\d{3,4}\([+-]?\d+\))', s)
    if m:
        return m.group(1)
    m2 = re.match(r'^(\d{3,4})', s)
    return m2.group(1) if m2 else s

def format_past_run_text(row: pd.Series, idx: int) -> str:
    """Geminiプロンプト用の近5走生ログ1走分テキスト整形"""
    r_raw = row.get(f'前走{idx}_着順', '')
    if pd.isna(r_raw) or str(r_raw).strip() in ['', 'nan', 'None']:
        return ""
    r_str = str(r_raw).strip()
    baba = str(row.get(f'前走{idx}_馬場', '-')).strip()
    if baba in ['nan', 'None', '']: baba = "-"
    pass_str = str(row.get(f'前走{idx}_通過順', '-')).strip()
    if pass_str in ['nan', 'None', '']: pass_str = "-"
    l3_str = str(row.get(f'前走{idx}_上がり3F', '-')).strip()
    if l3_str in ['nan', 'None', '']: l3_str = "-"
    t_str = str(row.get(f'前走{idx}_タイム', '-')).strip()
    if t_str in ['nan', 'None', '']: t_str = "-"
    bw = row.get(f'前走{idx}_馬体重', np.nan)
    bw_diff = row.get(f'前走{idx}_体重増減', np.nan)
    bw_txt = ""
    if pd.notna(bw):
        try:
            bw_txt = f"/馬体:{int(float(bw))}({int(float(bw_diff)):+d})" if pd.notna(bw_diff) else f"/馬体:{int(float(bw))}"
        except:
            pass
    return f"前走{idx}:{r_str}着[馬場:{baba}/時計:{t_str}/通過:{pass_str}/上3F:{l3_str}{bw_txt}]"

def get_badge_class(mark):
    if pd.isna(mark): return "badge-keshi"
    if "◎" in mark: return "badge-honmei"
    elif "◯" in mark: return "badge-taikou"
    elif "▲" in mark: return "badge-tana"
    elif "△" in mark: return "badge-renka"
    elif "☆" in mark: return "badge-ana"
    elif "注" in mark: return "badge-chuu"
    return "badge-keshi"

def get_frame_badge_html(wakuban: Any) -> str:
    try: w = int(float(wakuban))
    except: return "<span class='waku-badge' style='background:#eee; color:#333;'>-</span>"
    colors = {
        1: ("#ffffff", "#000000", "1px solid #bbb"), 2: ("#222222", "#ffffff", "none"),
        3: ("#e74c3c", "#ffffff", "none"), 4: ("#3498db", "#ffffff", "none"),
        5: ("#f1c40f", "#222222", "none"), 6: ("#2ecc71", "#ffffff", "none"),
        7: ("#e67e22", "#ffffff", "none"), 8: ("#fd79a8", "#ffffff", "none"),
    }
    bg, fg, border = colors.get(w, ("#eee", "#333", "none"))
    return f"<span class='waku-badge' style='background:{bg}; color:{fg}; border:{border};'>{w}</span>"

def analyze_horse_physics_tags(row: pd.Series, course_info: Dict[str, Any], total_horses: int, track_cond: str = "良") -> Tuple[str, bool, bool]:
    """
    Python側で近走生ログ・枠順・脚質・斤量・道悪実績を解析し、確定物理タグを生成（Gemini補正用）
    戻り値: (タグ文字列, 特大レバレッジ許可フラグ, 3番手▲最低保証昇格フラグ)
    """
    tags = []
    has_special = False
    promote_top3 = False

    try: waku = int(float(row.get('枠番', 0)))
    except: waku = 0
    kyaku = str(row.get('公式脚質', row.get('脚質_アプリ用', '-'))).strip()
    fav_waku = course_info.get("fav_waku", [])
    fav_style = course_info.get("fav_style", [])

    is_fav_waku = (waku > 0 and waku in fav_waku)
    if is_fav_waku and kyaku in fav_style:
        tags.append("🔥【コース枠ドンピシャ】")
        has_special = True
    elif is_fav_waku:
        tags.append("【コース有利枠】")
    elif fav_waku and waku > 0 and waku not in fav_waku and len(fav_waku) <= 4:
        if (min(fav_waku) >= 5 and waku <= 2) or (max(fav_waku) <= 4 and waku >= 7):
            tags.append("⚠️【コース逆バイアス枠】")

    # 前走1・前走2から「着順詐欺判定」をチェック
    has_chakujun_sagi = False
    for idx in [1, 2]:
        r_raw = str(row.get(f'前走{idx}_着順', ''))
        m_rank = re.search(r'(\d+)', r_raw)
        p_rank = int(m_rank.group(1)) if m_rank else 99

        p_pass = str(row.get(f'前走{idx}_通過順', ''))
        pass_nums = [int(x) for x in re.findall(r'\d+', p_pass)]
        first_c = pass_nums[0] if pass_nums else None
        last_c = pass_nums[-1] if pass_nums else None

        try: p_agari = float(str(row.get(f'前走{idx}_上がり3F', '')))
        except: p_agari = 99.0

        if p_rank >= 6 and last_c is not None and last_c >= max(7, int(total_horses * 0.6)) and p_agari <= 38.8:
            tags.append(f"🔥【着順詐欺判定({idx}走前)：後方から脚余し】")
            has_special = True
            has_chakujun_sagi = True
            break
        elif p_rank >= 6 and first_c is not None and first_c <= 2:
            tags.append(f"🔥【着順詐欺判定({idx}走前)：ハイペース先行潰れ】")
            has_special = True
            has_chakujun_sagi = True
            break

    # 有利枠 × 着順詐欺 の複合特注タグ（馬連3BOXの▲昇格対象）
    if is_fav_waku and has_chakujun_sagi:
        tags.append("🔥【有利枠×着順詐欺：特注穴馬】")
        promote_top3 = True

    # 重・不良馬場時の「道悪巧者タグ」（近3走以内に重・不良で3着以内好走）
    if track_cond in ["重", "不良"]:
        for idx in [1, 2, 3]:
            baba_past = str(row.get(f'前走{idx}_馬場', '')).strip()
            r_raw = str(row.get(f'前走{idx}_着順', ''))
            m_rank = re.search(r'(\d+)', r_raw)
            p_rank = int(m_rank.group(1)) if m_rank else 99
            if baba_past in ["重", "不良"] and p_rank <= 3:
                tags.append(f"🔥【道悪巧者：{idx}走前{baba_past}{p_rank}着】")
                has_special = True
                promote_top3 = True
                break

    try: kinryo = float(row.get('斤量', 55.0))
    except: kinryo = 55.0
    if kinryo <= 51.0:
        tags.append("【減量恩恵：51kg以下軽量】")
        has_special = True
    elif kinryo >= 58.0:
        tags.append("⚠️【酷量58kg以上】")

    tag_str = " ".join(tags) if tags else "特筆物理タグなし"
    return tag_str, has_special, promote_top3

def save_prediction_history(merged_df: pd.DataFrame, race_meta: Dict[str, Any], bet_summary: str, ken_reason: str):
    """Gemini補正後の印・スコア・買い目を prediction_history_chiho.csv に自動追記保存"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_rows = []
    for _, r in merged_df.iterrows():
        save_rows.append({
            "saved_at": now_str,
            "date": race_meta.get("date", ""),
            "race_id": race_meta.get("race_id", ""),
            "place_name": race_meta.get("place_name", ""),
            "r_num": race_meta.get("r_num", ""),
            "race_name": race_meta.get("race_name", ""),
            "distance": race_meta.get("distance", ""),
            "track_condition": race_meta.get("track_condition", ""),
            "ken_flag": "見送り" if ken_reason else "勝負",
            "ken_reason": ken_reason if ken_reason else "-",
            "recommended_bet": bet_summary,
            "枠番": r.get("枠番", ""),
            "馬番": int(float(r.get("馬番", 0))) if pd.notna(r.get("馬番")) else 0,
            "馬名": r.get("馬名", ""),
            "騎手": r.get("騎手", ""),
            "公式脚質": r.get("公式脚質", r.get("脚質_アプリ用", "-")),
            "人気": r.get("人気", np.nan),
            "オッズ": r.get("オッズ", np.nan),
            "初期印": r.get("初期印", "消"),
            "base_prob": round(float(r.get("win_prob", 0.0)) * 100, 2),
            "gemini_coeff": round(float(r.get("補正係数", 1.0)), 2),
            "final_prob": round(float(r.get("final_prob", 0.0)) * 100, 2),
            "最終印": r.get("最終印", "消"),
            "短評": r.get("短評", "")
        })
    df_new = pd.DataFrame(save_rows)
    if os.path.exists(HISTORY_CSV):
        try:
            df_old = pd.read_csv(HISTORY_CSV, dtype={"race_id": str}, encoding="utf-8-sig")
            mask_keep = ~((df_old["race_id"].astype(str) == str(race_meta.get("race_id", ""))) &
                          (df_old["track_condition"].astype(str) == str(race_meta.get("track_condition", ""))))
            df_out = pd.concat([df_old[mask_keep], df_new], ignore_index=True)
        except Exception:
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

@st.cache_data
def load_future_data() -> pd.DataFrame:
    if not os.path.exists(FUTURE_CSV): return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try:
            df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str, '枠番': str, '馬番': str}, low_memory=False, encoding=enc)
            if not df.empty:
                df['race_id'] = pd.to_numeric(df['race_id'], errors='coerce').fillna(0).astype(np.int64).astype(str)
                df['place_code'] = df['race_id'].astype(str).str[4:6]
                df['place_name'] = df.get('place_name', '地方')
                df['r_num'] = pd.to_numeric(df['r_num'], errors='coerce').fillna(1).astype(int)
                df['day_label'] = df['date'].astype(str).str.strip() if 'date' in df.columns else "当日"
                df['馬名_clean'] = df['馬名'].astype(str).apply(clean_name)
                return df
        except: continue
    return pd.DataFrame()

@st.cache_resource
def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try: return joblib.load(CACHE_FILE)
        except: pass
    return {}

df_future = load_future_data()

# ==========================================
# 🎯 UIレンダリング関数（スッキリ表示版）
# ==========================================
def render_race_predictions(race_id_target, track_cond):
    if df_future.empty: return None
    cache_data = load_cache()
    if not cache_data:
        st.warning("⚠️ キャッシュファイル(app_cache_chiho.pkl)が見つかりません。create_cache_chiho.py を実行してください。")
        return None
    
    cached_item = cache_data.get(str(race_id_target))
    if not cached_item:
        st.warning(f"⚠️ 選択されたレース ({race_id_target}) の予測スコアがキャッシュ内に未登録です。create_cache_chiho.py を再実行してください。")
        return None

    if isinstance(cached_item, dict) and "records" in cached_item:
        df = pd.DataFrame(cached_item["records"])
    elif isinstance(cached_item, list):
        df = pd.DataFrame(cached_item)
    else: return None

    if df.empty: return None

    df['馬名_clean'] = df['馬名'].apply(clean_name)
    df_f = df_future[df_future['race_id'].astype(str) == str(race_id_target)].copy()
    
    target_cols = [c for c in df_f.columns if '前走' in c or 'オッズ' in c or '人気' in c or '性齢' in c or c == '馬体重']
    df = df.drop(columns=[c for c in target_cols if c in df.columns], errors='ignore')
    df = pd.merge(df, df_f[['馬名_clean'] + target_cols], on='馬名_clean', how='left')

    target_score_col = f'score_{track_cond}' if f'score_{track_cond}' in df.columns else 'score_disp_base'
    df['base_score'] = pd.to_numeric(df.get(target_score_col, df.get('score_disp_base', 70)), errors='coerce').fillna(70)
    
    exp_scores = np.exp((df['base_score'] - df['base_score'].max()) / 16.0)
    df['win_prob'] = exp_scores / exp_scores.sum()
    
    df = df.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)
    mark_list = ['◎', '◯', '▲', '△1', '△2']
    df['初期印'] = [mark_list[i] if i < len(mark_list) else '消' for i in range(len(df))]

    return df

def generate_base_table(disp_df, is_shinba=False):
    html = "<div class='table-container'><table class='kachi-table'>"
    init_prob_header = "Python参考値" if is_shinba else "Python勝率"
    init_mark_header = "初期印(参考)" if is_shinba else "初期印"
    html += (
        f"<tr><th>枠</th><th>馬番</th><th style='text-align:left;'>馬名</th>"
        f"<th>性齢</th><th>斤量</th><th>馬体重</th><th>騎手</th><th>脚質</th>"
        f"<th>人気(オッズ)</th><th>{init_prob_header}</th><th>{init_mark_header}</th></tr>"
    )
    for _, r in disp_df.iterrows():
        w_badge = get_frame_badge_html(r.get('枠番'))
        m_num = int(float(r.get('馬番', 0))) if pd.notna(r.get('馬番')) else 0
        win_val = float(r.get('win_prob', 0)) * 100
        kinryo_val = float(r.get('斤量', 55.0))
        bw_clean = format_body_weight_clean(r.get('馬体重'))
        mark = r.get('初期印', '消')
        kyaku = str(r.get('公式脚質', r.get('脚質_アプリ用', '-')))
        
        try:
            pop_val = int(float(r.get('人気', 0)))
            odds_val = float(r.get('オッズ', 0))
            pop_str = f"<b>{pop_val}</b>人気<br><span style='font-size:0.85em; color:#666;'>({odds_val:.1f}倍)</span>" if pop_val > 0 else "<span style='color:#ccc;'>-</span>"
        except:
            pop_str = "<span style='color:#ccc;'>-</span>"

        prob_disp = f"<span style='color:#7f8c8d;'>({win_val:.1f}%)</span>" if is_shinba else f"<b>{win_val:.1f}%</b>"
        mark_disp = "<span class='badge-mark badge-keshi'>-</span>" if is_shinba else f"<span class='badge-mark {get_badge_class(mark)}'>{mark}</span>"

        html += (
            f"<tr><td>{w_badge}</td><td><b>{m_num:02d}</b></td>"
            f"<td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
            f"<td>{r.get('性齢', '-')}</td><td>{kinryo_val:.1f}kg</td><td>{bw_clean}</td>"
            f"<td>{r.get('騎手', '-')}</td><td>{kyaku}</td><td>{pop_str}</td>"
            f"<td>{prob_disp}</td><td>{mark_disp}</td></tr>"
        )
    html += "</table></div>"
    return html

def generate_fusion_multiplied_table(merged_df, is_shinba=False):
    html = "<div class='table-container'><table class='kachi-table'>"
    if is_shinba:
        html += "<tr><th>枠</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>人気(オッズ)</th><th>Gemini評価値</th><th>最終印</th><th style='text-align:left; min-width: 420px;'>Gemini単独評価根拠 (コース枠・騎手適性)</th></tr>"
    else:
        html += "<tr><th>枠</th><th>初期印</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>人気(オッズ)</th><th>元勝率</th><th>Gemini倍率</th><th>補正後勝率</th><th>最終印</th><th style='text-align:left; min-width: 420px;'>補正理由 (近5走生ログ・物理バイアス・展開根拠)</th></tr>"
    
    for _, r in merged_df.iterrows():
        w_badge = get_frame_badge_html(r.get('枠番'))
        m_num = int(float(r.get('馬番', 0))) if pd.notna(r.get('馬番')) else 0
        
        f_mark = r.get('最終印', '消')
        b_mark = r.get('初期印', '消')
        w_base = float(r.get('win_prob', 0)) * 100
        w_final = float(r.get('final_prob', 0)) * 100
        mult = float(r.get('補正係数', 1.0))
        
        if mult >= 1.2:
            mult_str = f"<span class='multiplier-up'>×{mult:.2f} ⬆</span>" if not is_shinba else f"<span class='multiplier-up'>★ {mult:.2f}</span>"
        elif mult <= 0.8:
            mult_str = f"<span class='multiplier-down'>×{mult:.2f} ⬇</span>" if not is_shinba else f"<span class='multiplier-down'>▼ {mult:.2f}</span>"
        else:
            mult_str = f"<span class='multiplier-flat'>×{mult:.2f}</span>" if not is_shinba else f"<span class='multiplier-flat'>- {mult:.2f}</span>"
            
        tanpyo = r.get('短評', '-')

        try:
            pop_val = int(float(r.get('人気', 0)))
            odds_val = float(r.get('オッズ', 0))
            pop_str = f"<b>{pop_val}</b>人気 ({odds_val:.1f}倍)" if pop_val > 0 else "-"
        except:
            pop_str = "-"
            
        if is_shinba:
            html += f"<tr><td>{w_badge}</td><td><b>{m_num:02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
            html += f"<td>{pop_str}</td><td>{mult_str}</td>"
            html += f"<td><span class='badge-mark {get_badge_class(f_mark)}'>{f_mark}</span></td>"
            html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 10px; color:#2c3e50;'>{tanpyo}</td></tr>"
        else:
            html += f"<tr><td>{w_badge}</td><td><span class='badge-mark {get_badge_class(b_mark)}'>{b_mark}</span></td>"
            html += f"<td><b>{m_num:02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
            html += f"<td>{pop_str}</td><td>{w_base:.1f}%</td><td>{mult_str}</td><td><b style='color:#e74c3c; font-size:16px;'>{w_final:.1f}%</b></td>"
            html += f"<td><span class='badge-mark {get_badge_class(f_mark)}'>{f_mark}</span></td>"
            html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 10px; color:#2c3e50;'>{tanpyo}</td></tr>"
    html += "</table></div>"
    return html

def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()

# ==========================================
# 🎯 メイン画面ロジック
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=clear_all_caches, use_container_width=True)
if os.path.exists(HISTORY_CSV):
    with open(HISTORY_CSV, "rb") as f_hist:
        st.sidebar.download_button(
            label="📥 予測履歴CSVをダウンロード",
            data=f_hist,
            file_name=HISTORY_CSV,
            mime="text/csv",
            use_container_width=True
        )

if df_future.empty:
    st.warning("出馬表データ (future_races_chiho.csv) が見つかりません。")
    st.stop()

st.markdown("<div class='section-header' style='margin-top: 0;'>🎯 レース選択</div>", unsafe_allow_html=True)
sel_date = st.radio("開催日", sorted(df_future['day_label'].unique()), horizontal=True, label_visibility="collapsed")
day_df = df_future[df_future['day_label'] == sel_date]
places = day_df['place_name'].unique()

place_tabs = st.tabs([f"📍 {p}" for p in places])
for i, place in enumerate(places):
    with place_tabs[i]:
        place_df = day_df[day_df['place_name'] == place]
        for j in range(0, len(place_df['r_num'].unique()), 6):
            cols = st.columns(6)
            for k, r in enumerate(sorted(place_df['r_num'].unique())[j:j+6]):
                r_id = str(place_df[place_df['r_num'] == r]['race_id'].iloc[0])
                if cols[k].button(f"{r}R", key=f"btn_{r_id}", use_container_width=True):
                    st.session_state['selected_race_id'] = r_id

if st.session_state.get('selected_race_id'):
    t_id = str(st.session_state['selected_race_id'])
    r_info = df_future[df_future['race_id'].astype(str) == t_id].iloc[0]
    
    csv_baba_raw = str(r_info.get('track_condition', '良')).strip()
    baba_norm_map = {'良': '良', '稍': '稍重', '稍重': '稍重', '重': '重', '不': '不良', '不良': '不良'}
    csv_baba = baba_norm_map.get(csv_baba_raw, '良')
    baba_options = ["良", "稍重", "重", "不良"]
    default_baba_idx = baba_options.index(csv_baba) if csv_baba in baba_options else 0
    track_condition = st.sidebar.radio("想定馬場", options=baba_options, index=default_baba_idx)

    st.markdown("---")
    res_df = render_race_predictions(t_id, track_condition)
    if res_df is None: st.stop()
    
    place_name_val = r_info.get('place_name', '地方')
    dist_val = r_info.get('distance', 1400)
    race_name_str = str(r_info.get('race_name', ''))
    course_info = get_course_physics_info(place_name_val, dist_val)
    course_physics_desc = course_info["summary"]

    has_no_past = res_df['前走1_着順'].isna().all() or (res_df['前走1_着順'].astype(str).str.strip() == '').all()
    is_shinba = ("新馬" in race_name_str) or has_no_past
    is_age2 = (not is_shinba) and ("2歳" in race_name_str or "２歳" in race_name_str or res_df['性齢'].astype(str).str.contains('2').all())

    if is_shinba:
        mode_badge_html = "<span class='shinba-badge'>🐴 新馬・初出走戦 (見送り推奨)</span>"
        active_system_instruction = base_system_instruction + "\n" + shinba_addon
        mode_label = "新馬・初出走モード"
    elif is_age2:
        mode_badge_html = "<span class='age2-badge'>🌱 2歳若馬戦モード (成長・枠バイアス重視)</span>"
        active_system_instruction = base_system_instruction + "\n" + age2_addon
        mode_label = "2歳若馬戦モード"
    else:
        mode_badge_html = "<span class='old-badge'>🔥 古馬・上級戦モード (物理バイアス・近5走重視)</span>"
        active_system_instruction = base_system_instruction + "\n" + old_horse_addon
        mode_label = "古馬・上級戦モード"
    
    st.markdown(f"<h2>🚀 {place_name_val} {r_info['r_num']}R 【{race_name_str}】 {mode_badge_html} <span style='font-size:18px; color:#7f8c8d;'>({r_info.get('surface','ダ')}{dist_val}m / 想定馬場: {track_condition})</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='course-banner'>🧭 コース物理: {course_physics_desc}</div>", unsafe_allow_html=True)
    
    if is_shinba:
        st.info("💡【新馬・初出走モード】過去走データが存在しないため、Pythonの初期勝率は参考外としています（原則見送り推奨）。")
        st.markdown("<div class='section-header'>📊 1. 出走馬一覧（過去走データなし）</div>", unsafe_allow_html=True)
        st.markdown(generate_base_table(res_df, is_shinba=True), unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-header'>📊 1. Pythonベース地力スコア (基礎勝率)</div>", unsafe_allow_html=True)
        st.markdown(generate_base_table(res_df, is_shinba=False), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(f"⚡ Geminiによる精密レバレッジ補正を実行（{mode_label}）", type="primary", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("サイドバーに GEMINI_API_KEY を入力してください。")
            st.stop()

        total_horses = len(res_df)
        nige_count = int((res_df.get('公式脚質', res_df.get('脚質_アプリ用', '')).astype(str) == '逃').sum())

        # 裏側で近5走生ログ・枠順・脚質・斤量・道悪実績から物理タグを生成（Gemini補正用）
        physics_tags_list = []
        special_flag_map = {}
        promote_top3_map = {}
        for _, r in res_df.iterrows():
            tag_str, has_sp, prom_3 = analyze_horse_physics_tags(r, course_info, total_horses, track_condition)
            physics_tags_list.append(tag_str)
            special_flag_map[r['馬名_clean']] = has_sp
            promote_top3_map[r['馬名_clean']] = prom_3
        res_df['physics_tags'] = physics_tags_list

        # Geminiプロンプトへ近5走の着順・馬場・通過順・上がり3F・タイム・馬体重増減を全量投入
        table_summary = []
        for _, r in res_df.iterrows():
            past_texts = [format_past_run_text(r, j) for j in range(1, 6)]
            past_texts = [pt for pt in past_texts if pt]
            p_history = " / ".join(past_texts) if past_texts else "近走データなし"
            
            bw_clean = format_body_weight_clean(r.get('馬体重'))
            prob_disp_str = "Python勝率:(新馬戦のため無効)" if is_shinba else f"Python勝率:{float(r.get('win_prob',0))*100:.1f}%"
            info = (
                f"枠:{r.get('枠番')} | 馬番:{int(float(r.get('馬番',0))):02d} | 馬名:{r.get('馬名','')} | "
                f"{prob_disp_str} | 斤量:{r.get('斤量','-')}kg | 馬体重:{bw_clean} | "
                f"脚質:{r.get('公式脚質', r.get('脚質_アプリ用', '-'))} | 騎手:{r.get('騎手','-')} | "
                f"物理判定:{r['physics_tags']} | 【近5走生ログ】{p_history}"
            )
            table_summary.append(info)

        prompt = f"""
以下の出走馬データ（近5走生ログ含む）に基づき、指示に従って全出走馬の補正係数（0.50〜1.60）と理由を出力してください。

【適用評価モード】: {mode_label}
【対象レース】: {place_name_val} {r_info['r_num']}R 【{race_name_str}】 ({r_info.get('surface','ダ')}{dist_val}m / 想定馬場:{track_condition})
【コース特性】: {course_physics_desc}
【出走頭数 / 逃げ馬数】: 全{total_horses}頭 / 逃げ脚質:{nige_count}頭

【出走馬データ一覧】
{chr(10).join(table_summary)}
"""

        with st.spinner(f"🧠 Geminiが【{mode_label}】で地方ダートの物理バイアス・近5走生ログから精密鑑定中..."):
            client = genai.Client(api_key=GEMINI_API_KEY)
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=prompt, 
                    config=types.GenerateContentConfig(
                        system_instruction=active_system_instruction, 
                        temperature=0.1, 
                        response_mime_type="application/json",
                        response_schema=RaceCoefficientResult,
                    )
                )
                
                result = RaceCoefficientResult.model_validate_json(response.text)
                coeff_df = pd.DataFrame([c.model_dump() for c in result.coefficients])
                
                if not coeff_df.empty and '馬名' in coeff_df.columns:
                    coeff_df['馬名_clean'] = coeff_df['馬名'].astype(str).apply(clean_name)
                    coeff_df_clean = coeff_df[['馬名_clean', '補正係数', '短評']].dropna(subset=['馬名_clean'])
                    
                    merged = pd.merge(res_df, coeff_df_clean, on='馬名_clean', how='left')
                    merged['補正係数'] = merged['補正係数'].fillna(1.0)
                    merged['短評'] = merged['短評'].fillna('データ保留')

                    # 🛡️ 防衛的クリッピング（インフレ防止）
                    clipped_coeffs = []
                    for _, r in merged.iterrows():
                        mult = float(r.get('補正係数', 1.0))
                        has_sp = special_flag_map.get(r['馬名_clean'], False)
                        p_tags = str(r.get('physics_tags', ''))

                        if not is_shinba and not has_sp and mult > 1.30:
                            mult = 1.30
                        if ("逆バイアス枠" in p_tags or "酷量" in p_tags) and mult > 1.00:
                            mult = 1.00
                        clipped_coeffs.append(mult)
                    merged['補正係数'] = clipped_coeffs
                    
                    if is_shinba:
                        merged['raw_final'] = merged['補正係数']
                    else:
                        merged['raw_final'] = merged['win_prob'] * merged['補正係数']
                    merged['final_prob'] = merged['raw_final'] / merged['raw_final'].sum()
                    merged = merged.sort_values(by='final_prob', ascending=False).reset_index(drop=True)

                    # 🔥 ルール修正②：道悪巧者（重・不良好走）または有利枠×着順詐欺の複合タグ持ちが4番手以下の場合、3番手(▲)へ自動引き上げ
                    if not is_shinba and len(merged) >= 4:
                        prom_candidates = [
                            idx for idx in range(3, len(merged))
                            if promote_top3_map.get(merged.iloc[idx]['馬名_clean'], False)
                            and "逆バイアス枠" not in str(merged.iloc[idx].get('physics_tags', ''))
                        ]
                        if prom_candidates:
                            best_prom_idx = prom_candidates[0]
                            row_prom = merged.iloc[[best_prom_idx]].copy()
                            row_prom['短評'] = "【▲自動昇格：道悪/有利枠×着順詐欺】" + row_prom['短評'].astype(str)
                            others = merged.drop(index=best_prom_idx).reset_index(drop=True)
                            merged = pd.concat([others.iloc[:2], row_prom, others.iloc[2:]], ignore_index=True)
                    
                    marks = ['◎', '◯', '▲', '△1', '△2']
                    final_marks = ['消'] * len(merged)
                    for idx in range(min(len(marks), len(merged))):
                        final_marks[idx] = marks[idx]

                    chuu_candidates = []
                    for idx in range(len(marks), len(merged)):
                        row_m = merged.iloc[idx]
                        mult = float(row_m.get('補正係数', 1.0))
                        has_sp = special_flag_map.get(row_m['馬名_clean'], False)
                        if mult >= 1.38 or (has_sp and mult >= 1.30):
                            chuu_candidates.append((idx, mult))
                    chuu_candidates.sort(key=lambda x: -x[1])
                    for candidate in chuu_candidates[:2]:
                        final_marks[candidate[0]] = '注'

                    merged['最終印'] = final_marks
                    
                    st.markdown(f"""
                    <div class='gemini-view-card'>
                        <div class='gemini-view-title'>📝 トラックマンGeminiの展開・物理バイアス統括 ({mode_label})</div>
                        <p class='gemini-view-text'>{result.summary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ==========================================
                    # 🎟 実戦検証済み：見送り（ケン）フィルター ＆ 馬連3頭BOX・馬単最適化
                    # ==========================================
                    probs = merged['final_prob'].values
                    p1 = probs[0] if len(probs) > 0 else 0.0
                    p4 = probs[3] if len(probs) > 3 else 0.0

                    h_o = merged.iloc[0]
                    h_m = merged.iloc[1] if len(merged) > 1 else h_o
                    h_t = merged.iloc[2] if len(merged) > 2 else h_m

                    try: pop_o = int(float(h_o.get('人気', 0)))
                    except: pop_o = 0
                    try: pop_m = int(float(h_m.get('人気', 0)))
                    except: pop_m = 0
                    try: pop_t = int(float(h_t.get('人気', 0)))
                    except: pop_t = 0

                    ken_reasons = []
                    if is_shinba:
                        ken_reasons.append("新馬・初出走戦（過去走データなし）")
                    if pop_o >= 3:
                        ken_reasons.append(f"◎（{h_o['馬名']}）が{pop_o}番人気（3番人気以下は期待値低下）")
                    if not is_shinba and (p1 - p4) < 0.065:
                        ken_reasons.append("大混戦レース（上位4頭の勝率差が6.5%未満で軸不安定）")

                    ken_reason_str = " / ".join(ken_reasons)

                    chuu_horses = merged[merged['最終印'] == '注']
                    chuu_html = ""
                    if not chuu_horses.empty:
                        medals = ['🥇', '🥈']
                        chuu_list = []
                        for idx_c, (_, ch) in enumerate(chuu_horses.iterrows()):
                            medal = medals[idx_c] if idx_c < len(medals) else "・"
                            score_lbl = f"スコア {ch['補正係数']:.2f}" if is_shinba else f"倍率 ×{ch['補正係数']:.2f}"
                            chuu_list.append(f"<b>{medal}【馬番{int(float(ch['馬番'])):02d} {ch['馬名']}】</b>({score_lbl}) : {ch.get('短評', '-')}")
                        chuu_html = f"<div style='margin-top: 12px; padding: 10px 14px; background-color: #fff9e6; border-left: 4px solid #f39c12; border-radius: 4px;'><div style='font-weight: bold; color: #d35400; font-size: 15px;'>🔥【物理バイアス・着順詐欺の特注伏兵（注）】</div><div style='font-size: 14px; color: #444; margin-top: 4px; line-height: 1.6;'>{'<br>'.join(chuu_list)}</div></div>"

                    m_o = f"{int(float(h_o['馬番'])):02d}"
                    m_m = f"{int(float(h_m['馬番'])):02d}"
                    m_t = f"{int(float(h_t['馬番'])):02d}"

                    if ken_reason_str:
                        card_cls = "sense-card-ken"
                        title_color = "#c53030"
                        judge_title = "🚫 見送り推奨（ケン）判定"
                        main_bet_txt = f"見送り推奨（理由: {ken_reason_str}）"
                        sub_bet_txt = f"参考買い目（少額遊び用）: 馬連 3頭BOX {m_o} - {m_m} - {m_t} (3点)"
                        bet_log_str = f"【ケン】{main_bet_txt}"
                    elif (1 <= pop_o <= 2) and (pop_m >= 3 or pop_t >= 4):
                        card_cls = "sense-card-hot"
                        title_color = "#b7791f"
                        judge_title = f"🔥 激アツ勝負レース（◎{pop_o}人気 × ◯{pop_m}人気 × ▲{pop_t}人気 中穴妙味あり）"
                        main_bet_txt = f"<b>馬連 3頭BOX（3点）</b> ➔ <b>{m_o} ({h_o['馬名']}) ・ {m_m} ({h_m['馬名']}) ・ {m_t} ({h_t['馬名']})</b>"
                        sub_bet_txt = f"<b>馬単 1着固定（4点）</b> ➔ {m_o}(◎) ➔ {m_m}, {m_t}, △1, △2 ／ <b>単勝</b> ➔ {m_o}(◎), {m_m}(◯) 2点"
                        bet_log_str = f"【激アツ勝負】馬連3BOX: {m_o}-{m_m}-{m_t} (3点) / サブ馬単: {m_o}->相手4頭"
                    else:
                        card_cls = "sense-card"
                        title_color = "#8e44ad"
                        judge_title = f"✅ 標準勝負レース（上位人気決着濃厚・ガミり注意）"
                        main_bet_txt = f"<b>馬連 3頭BOX（3点）</b> ➔ <b>{m_o} ({h_o['馬名']}) ・ {m_m} ({h_m['馬名']}) ・ {m_t} ({h_t['馬名']})</b>（※オッズ安めなら見送り or 資金配分）"
                        sub_bet_txt = f"<b>馬単 1着固定（4点）</b> ➔ {m_o}(◎) ➔ {m_m}, {m_t}, △1, △2"
                        bet_log_str = f"【標準勝負】馬連3BOX: {m_o}-{m_m}-{m_t} (3点) / 馬単: {m_o}->相手4頭"

                    st.markdown(f"""
                    <div class='{card_cls}'>
                        <div class='sense-title' style='color:{title_color};'>{judge_title}</div>
                        <ul style='margin-top: 8px; margin-bottom: 10px; font-size: 15px; color: #333; line-height: 1.8;'>
                            <li><span class='strategy-badge-main'>🏆 メイン推奨</span> {main_bet_txt}</li>
                            <li><span class='strategy-badge-sub'>⚡ サブ・高配当</span> {sub_bet_txt}</li>
                        </ul>
                        <div style='margin-top: 10px; border-top: 1px dashed #ccc; padding-top: 8px; font-size: 14px; color: #444;'>
                            <b>【上位評価】</b> ◎ {m_o} {h_o['馬名']} ({h_o['final_prob']*100:.1f}%) ／ ◯ {m_m} {h_m['馬名']} ({h_m['final_prob']*100:.1f}%) ／ ▲ {m_t} {h_t['馬名']} ({h_t['final_prob']*100:.1f}%)
                        </div>
                        {chuu_html}
                    </div>
                    """, unsafe_allow_html=True)

                    # 📁 prediction_history_chiho.csv への自動保存
                    race_meta = {
                        "date": str(r_info.get("date", sel_date)),
                        "race_id": t_id,
                        "place_name": place_name_val,
                        "r_num": int(r_info.get("r_num", 1)),
                        "race_name": race_name_str,
                        "distance": dist_val,
                        "track_condition": track_condition,
                    }
                    save_prediction_history(merged, race_meta, bet_log_str, ken_reason_str)
                    st.toast(f"📁 {place_name_val}{r_info['r_num']}R の鑑定結果を {HISTORY_CSV} に自動保存しました！", icon="✅")
                    
                    section2_header = "🔥 2. Gemini単独評価結果（コース枠バイアス・騎手適性ソート）" if is_shinba else f"🔥 2. 数学融合結果（Python地力 × Gemini倍率 / {mode_label}）"
                    st.markdown(f"<div class='section-header'>{section2_header}</div>", unsafe_allow_html=True)
                    st.markdown(generate_fusion_multiplied_table(merged, is_shinba=is_shinba), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ API通信・計算エラー: {e}")
