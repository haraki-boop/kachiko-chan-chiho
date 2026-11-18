import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
from datetime import datetime, timedelta, timezone
import logging
from typing import List, Dict, Optional, Any

# ==========================================
# ロギング設定・定数
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NAR_PLACES: Dict[str, str] = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}
USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

def random_sleep(min_sec=0.8, max_sec=1.5):
    """サーバー負荷を抑える適度なインターバル"""
    time.sleep(random.uniform(min_sec, max_sec))

# ==========================================
# 通信基盤
# ==========================================
def create_retry_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def clean_text(text: Optional[str]) -> str:
    if not text: return ""
    return re.sub(r'[\s\u3000]+', '', str(text)).strip()

def clean_horse_name(text: Optional[str]) -> str:
    if not text: return ""
    s = re.sub(r'[\s\u3000]+', '', str(text)).strip()
    s = re.sub(r'\(.*?\)|\（.*?\）|［.*?］|\[.*?\]', '', s)
    return s

# ==========================================
# 近5走データ取得（馬柱）
# ==========================================
def get_past_data(session: requests.Session, race_id: str) -> Dict[str, Dict[str, str]]:
    url = f"https://nar.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    past_dict = {}
    
    def parse_past_cell(past_td) -> Dict[str, str]:
        data = {"着順": "", "馬場": "", "通過順": "", "上がり3F": "", "タイム": "", "馬体重": "", "体重増減": ""}
        if not past_td: return data
        
        full_text = past_td.get_text(" ", strip=True)
        full_text = re.sub(r'[\s\u3000\xa0]+', ' ', full_text)
        
        # 1. 着順
        num_span = past_td.find(class_=re.compile(r"Num", re.I))
        if num_span:
            data["着順"] = clean_text(num_span.text)
        else:
            m_rank = re.search(r'\b([1-9]|1[0-8])\b(?:\s+(?:芝|ダ|障))?', full_text)
            if m_rank: data["着順"] = m_rank.group(1)
            
        # 2. 馬場（独立抽出）
        m_baba = re.search(r'(良|稍重|稍|重|不良)', full_text)
        if m_baba:
            b_val = m_baba.group(1)
            data["馬場"] = "稍重" if b_val == "稍" else b_val
            
        # 3. 上がり3F
        m_up3f = re.search(r'\((\d{2}\.\d)\)', full_text)
        up3f_val = m_up3f.group(1) if m_up3f else ""
        if up3f_val:
            data["上がり3F"] = up3f_val

        # 4. タイム（独立抽出）
        m_time = re.search(r'\b(\d{1,2}[:.]\d{2}\.\d|\d{2}\.\d)\b', full_text)
        if m_time:
            t_val = m_time.group(1)
            if t_val != up3f_val:
                data["タイム"] = t_val
                
        # 5. 馬体重・体重増減
        m_weight = re.search(r'(\d{3,4})\s*\(\s*([+-]?\d+)\s*\)', full_text)
        if m_weight:
            data["馬体重"] = m_weight.group(1)
            data["体重増減"] = m_weight.group(2)
            
        # 6. 通過順
        m_pass = re.search(r'\b(\d+(?:-\d+)+)\b', full_text)
        if m_pass: data["通過順"] = m_pass.group(1)
        
        return data

    try:
        res = session.get(url, timeout=8)
        res.encoding = 'EUC-JP'
        soup = BeautifulSoup(res.content, 'html.parser')
        
        for tr in soup.find_all("tr", class_=re.compile(r"HorseList", re.I)):
            a_tag = tr.find("a", href=re.compile(r"/horse/"))
            if not a_tag: continue
            h_name = clean_horse_name(a_tag.text)
            
            horse_data = {"公式脚質": ""}
            for elem in tr.find_all(["span", "div", "i", "em", "b"]):
                txt = elem.get_text(strip=True)
                if txt in ["逃", "先", "差", "追"]:
                    horse_data["公式脚質"] = txt
                    break

            for i in range(1, 6):
                past_td = tr.find("td", class_=re.compile(rf"Past0?{i}|Past_{i}", re.I))
                if not past_td:
                    tds_with_races = [td for td in tr.find_all("td") if re.search(r'(?:芝|ダ|障)\s*\d{3,4}', td.get_text())]
                    if i <= len(tds_with_races):
                        past_td = tds_with_races[i-1]
                
                parsed = parse_past_cell(past_td)
                for k, v in parsed.items():
                    horse_data[f"前走{i}_{k}"] = v

            past_dict[h_name] = horse_data
            
    except Exception as e:
        logger.warning(f"馬柱取得失敗 (RaceID: {race_id}): {e}")
    return past_dict

# ==========================================
# メイン処理
# ==========================================
def get_today_chiho_races() -> pd.DataFrame:
    JST = timezone(timedelta(hours=+9), 'JST')
    today_dt = datetime.now(JST)
    
    date_str_hyph = today_dt.strftime("%Y-%m-%d")
    date_str_raw = today_dt.strftime("%Y%m%d")
    year = today_dt.strftime("%Y")
    mmdd = today_dt.strftime("%m%d")
    
    logger.info(f"本日の地方競馬データを取得中... ({date_str_hyph})")
    
    session = create_retry_session()
    url_list = f"https://nar.netkeiba.com/top/race_list.html?kaisai_date={date_str_raw}"
    active_places = set()
    
    try:
        res = session.get(url_list, timeout=8)
        res.encoding = 'EUC-JP'
        soup = BeautifulSoup(res.content, 'html.parser')
        pattern = rf'race_id={year}(\d{{2}}){mmdd}'
        for a in soup.find_all('a', href=re.compile(pattern)):
            match = re.search(pattern, a['href'])
            if match: active_places.add(match.group(1))
    except Exception:
        pass
    
    if not active_places:
        active_places = set(NAR_PLACES.keys())

    all_races: List[Dict[str, Any]] = []

    for place_code in active_places:
        place_name = NAR_PLACES.get(place_code, "地方")
        logger.info(f"📍 {place_name}競馬場の開催データを取得中...")
        
        for r_num in range(1, 13):
            race_id = f"{year}{place_code}{mmdd}{r_num:02d}"
            
            logger.info(f"   💬 {r_num}R 馬柱(近5走)データを収集...")
            past_dict = get_past_data(session, race_id)
            random_sleep(0.5, 1.0)
            
            url_shutuba = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
            try:
                res = session.get(url_shutuba, timeout=8)
                if res.status_code != 200: continue
                res.encoding = 'EUC-JP'
                soup = BeautifulSoup(res.content, 'html.parser')
                table = soup.find("table", class_=re.compile("RaceTable01|Shutuba_Table", re.I))
                if not table: continue

                race_name_tag = soup.find("div", class_="RaceName")
                race_name = clean_text(race_name_tag.text) if race_name_tag else f"{r_num}R"
                
                # 🚨 今回の距離・馬場・サーフェス（芝ダ）を取得
                distance = 1400
                surface = "ダ"
                track_cond = "良"
                data_intro = soup.find("div", class_="RaceData01") or soup.find("div", class_="RaceData00")
                if data_intro:
                    intro_text = data_intro.text
                    if "芝" in intro_text: surface = "芝"
                    elif "ダ" in intro_text or "ダート" in intro_text: surface = "ダ"
                    
                    dist_match = re.search(r'(\d{3,4})m', intro_text)
                    if dist_match: distance = int(dist_match.group(1))
                    
                    t_match = re.search(r'馬場:([良稍重不良]+)', intro_text)
                    if t_match: track_cond = t_match.group(1)

                for row in table.find_all("tr", class_=re.compile("HorseList", re.I)):
                    cols = row.find_all("td")
                    if len(cols) < 8: continue
                    
                    wakuban = clean_text(cols[0].text)
                    umaban = clean_text(cols[1].text)
                    if not umaban.isdigit(): continue 
                    
                    raw_h_name = clean_text(cols[3].text)
                    if not raw_h_name: continue
                    horse_name = clean_horse_name(raw_h_name)
                    
                    sei_rei = clean_text(cols[4].text)
                    kinryo = clean_text(cols[5].text) or "54.0"
                    jockey = clean_text(cols[6].text)
                    trainer = clean_text(cols[7].text)
                    
                    # 当日の馬体重とオッズ
                    weight = "470(0)"
                    if len(cols) > 8:
                        w_txt = clean_text(cols[8].text)
                        if re.search(r'\d{3}\([+-]?\d+\)', w_txt): weight = w_txt
                        
                    odds = clean_text(cols[9].text) if len(cols) > 9 and clean_text(cols[9].text) not in ["---", "", "0.0", "＊＊", "**"] else "15.0"
                    pop = clean_text(cols[10].text) if len(cols) > 10 and clean_text(cols[10].text) not in ["---", "", "0", "＊＊", "**"] else "99"

                    # 前走のデフォルト設定
                    default_p_data = {"公式脚質": ""}
                    for i in range(1, 6):
                        for k in ["着順", "馬場", "通過順", "上がり3F", "タイム", "馬体重", "体重増減"]:
                            default_p_data[f"前走{i}_{k}"] = ""
                    p_data = past_dict.get(horse_name, default_p_data)

                    record = {
                        "date": date_str_hyph, "race_id": str(race_id), "place_name": place_name,
                        "r_num": r_num, "race_name": race_name, "distance": distance,
                        "surface": surface, "track_condition": track_cond,  # 🚨 追加: AIに必要な必須特徴量
                        "枠番": wakuban, "馬番": umaban, "馬名": horse_name, "性齢": sei_rei,
                        "斤量": kinryo, "騎手": jockey, "調教師": trainer, "馬体重": weight,
                        "オッズ": odds, "人気": pop,
                        "公式脚質": p_data.get("公式脚質", "")
                    }
                    
                    # 前走データの統合
                    for i in range(1, 6):
                        for k in ["着順", "馬場", "通過順", "上がり3F", "タイム", "馬体重", "体重増減"]:
                            record[f"前走{i}_{k}"] = p_data.get(f"前走{i}_{k}", "")

                    all_races.append(record)
                random_sleep(0.5, 1.2)
            except Exception as e:
                logger.error(f"   ⚠️ {r_num}R 取得エラー: {e}")
                continue

    return pd.DataFrame(all_races)

if __name__ == "__main__":
    df = get_today_chiho_races()
    if not df.empty:
        df.to_csv("future_races_chiho.csv", index=False, encoding='utf-8-sig')
        logger.info(f"✨ 成功: {df.shape[0]} 件の出馬表データ（芝ダ・馬場状態・近5走完全版）を保存しました！")
    else:
        logger.error("🚨 エラー: 本日の出馬表データが1件も取得できませんでした。")