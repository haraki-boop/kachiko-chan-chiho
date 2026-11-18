import os
import re
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# 設定・定数（ばんえい:65 は完全除外）
# ==========================================
DB_FILE = "nar_keiba_database.csv"

# 平地地方競馬 全14場
NAR_PLACES: Dict[str, str] = {
    "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎",  # 南関
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀",  # 西日本
    "47": "笠松", "48": "名古屋", "46": "金沢",             # 中海
    "30": "門別", "35": "盛岡", "36": "水沢"               # 北日本
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
]

def create_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def clean_text(text: Optional[str]) -> str:
    if not text: return ""
    return re.sub(r'[\s\u3000]+', '', str(text)).strip()

def clean_horse_name(text: Optional[str]) -> str:
    if not text: return ""
    s = re.sub(r'[\s\u3000]+', '', str(text)).strip()
    return re.sub(r'\(.*?\)|\（.*?\）|［.*?］|\[.*?\]', '', s)

# ==========================================
# 取得済み日付のチェック（安全再開用）
# ==========================================
def get_start_date_and_status(csv_path: str, default_start: datetime) -> datetime:
    """
    勝手に削除はせず、ファイルが存在する場合は最終記録日の翌日を返す
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"📂 新規作成モード: {default_start.strftime('%Y-%m-%d')} から取得を開始します。")
        return default_start

    try:
        df_dates = pd.read_csv(csv_path, usecols=['date'], dtype={'date': str})
        valid_dates = pd.to_datetime(df_dates['date'].dropna(), errors='coerce').dropna()

        if valid_dates.empty:
            print(f"📂 有効な日付がありません。{default_start.strftime('%Y-%m-%d')} から開始します。")
            return default_start

        last_date = valid_dates.max()
        next_date = last_date + timedelta(days=1)
        print("==================================================")
        print(f"🔍 既存データベースを確認しました (総レコード数: {len(df_dates):,} 件)")
        print(f"📌 最終記録日: {last_date.strftime('%Y-%m-%d')}")
        print(f"➡️  未取得の 【{next_date.strftime('%Y-%m-%d')}】 から続きを追記します。")
        print("==================================================")
        return next_date

    except Exception as e:
        print(f"⚠️ ファイル読み込みエラー ({e})。指定開始日よりスタートします。")
        return default_start

# ==========================================
# 馬柱ページから「前走1〜5走」および「公式脚質」を取得（堅牢化版）
# ==========================================
def get_past_data(session: requests.Session, race_id: str) -> Dict[str, Dict[str, str]]:
    url = f"https://nar.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    past_dict = {}

    def parse_past_cell(past_td) -> Dict[str, str]:
        data = {"着順": "", "馬場": "", "通過順": "", "上がり3F": "", "タイム": "", "馬体重": "", "体重増減": ""}
        if not past_td: return data

        full_text = past_td.get_text(" ", strip=True)
        # 空白文字を半角スペースに統一
        full_text = re.sub(r'[\s\u3000\xa0]+', ' ', full_text)

        # 1. 着順
        num_span = past_td.find(class_=re.compile(r"Num", re.I))
        if num_span:
            data["着順"] = clean_text(num_span.text)
        else:
            m_rank = re.search(r'\b([1-9]|1[0-8])\b(?:\s+(?:芝|ダ|障))?', full_text)
            if m_rank: data["着順"] = m_rank.group(1)

        # 2. 馬場（単独抽出：前後の文字順に依存しない）
        m_baba = re.search(r'(良|稍重|稍|重|不良)', full_text)
        if m_baba:
            b_val = m_baba.group(1)
            data["馬場"] = "稍重" if b_val == "稍" else b_val

        # 3. 上がり3F (先に判定してタイム誤認を防止)
        m_up3f = re.search(r'\((\d{2}\.\d)\)', full_text)
        up3f_val = m_up3f.group(1) if m_up3f else ""
        if up3f_val:
            data["上がり3F"] = up3f_val

        # 4. タイム（単独抽出：1:23.4 / 1.23.4 / 58.4 など全対応）
        m_time = re.search(r'\b(\d{1,2}[:.]\d{2}\.\d|\d{2}\.\d)\b', full_text)
        if m_time:
            t_val = m_time.group(1)
            # 上がり3Fの値そのものでないかチェック
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
        headers = {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://nar.netkeiba.com/"}
        res = session.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
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
    except Exception:
        pass

    return past_dict

# ==========================================
# 1日分の確定結果＆前走データ統合スクレイピング
# ==========================================
def scrape_full_daily_results(session: requests.Session, target_date: datetime) -> List[Dict[str, str]]:
    date_str_raw = target_date.strftime("%Y%m%d")
    date_str_hyph = target_date.strftime("%Y-%m-%d")
    year = target_date.strftime("%Y")
    mmdd = target_date.strftime("%m%d")

    daily_data = []

    for place_code, place_name in NAR_PLACES.items():
        check_id = f"{year}{place_code}{mmdd}01"
        check_url = f"https://nar.netkeiba.com/race/result.html?race_id={check_id}"
        headers = {"User-Agent": random.choice(USER_AGENTS), "Referer": "https://nar.netkeiba.com/"}

        try:
            res_c = session.get(check_url, headers=headers, timeout=6)
            if res_c.status_code != 200 or ("RaceTable" not in res_c.text and "ResultTable" not in res_c.text):
                continue
        except Exception:
            continue

        print(f"  🏇 [{date_str_hyph} {place_name}] 開催検知 ➔ 取得中...")

        for r_num in range(1, 13):
            race_id = f"{year}{place_code}{mmdd}{r_num:02d}"

            # 1. 馬柱取得
            past_dict = get_past_data(session, race_id)

            # 2. 確定結果取得
            url_res = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
            res = None
            for attempt in range(3):
                try:
                    res = session.get(url_res, headers=headers, timeout=8)
                    if res.status_code == 200:
                        break
                    elif res.status_code in [403, 429]:
                        wait_sec = 15 * (attempt + 1)
                        print(f"    ⚠️ 一時ブロック検知 ({res.status_code})。{wait_sec}秒待機...")
                        time.sleep(wait_sec)
                    elif res.status_code == 400:
                        break
                except requests.exceptions.RequestException:
                    time.sleep(3)

            if res is None or res.status_code != 200:
                continue

            try:
                soup = BeautifulSoup(res.content, 'html.parser')
                table = soup.find("table", class_=re.compile(r"(RaceTable|ResultTable)", re.I)) or soup.find("table")
                if not table:
                    continue

                # レース条件判定
                surface = "ダ"
                distance = 1400
                track_condition = "良"
                data_intro = soup.find("div", class_="RaceData01") or soup.find("div", class_="RaceData00")
                if data_intro:
                    intro_text = data_intro.text
                    if "芝" in intro_text:
                        surface = "芝"
                    elif "ダ" in intro_text or "ダート" in intro_text:
                        surface = "ダ"

                    m_dist = re.search(r'(\d{3,4})m', intro_text)
                    if m_dist: distance = int(m_dist.group(1))
                    m_track = re.search(r'馬場:([良稍重不良]+)', intro_text)
                    if m_track: track_condition = m_track.group(1)

                header_tr = table.find("tr")
                if not header_tr: continue
                th_elements = header_tr.find_all(["th", "td"])

                def find_col_idx(keywords: list, class_patterns: list) -> int:
                    for i, th in enumerate(th_elements):
                        c_name = " ".join(th.get("class", []))
                        for cp in class_patterns:
                            if re.search(cp, c_name, re.I):
                                return i
                        t_text = th.text.strip().replace("\n", "").replace(" ", "")
                        for kw in keywords:
                            if kw in t_text:
                                return i
                    return -1

                idx_chaku = find_col_idx(["着順"], [r"Rank"])
                idx_waku  = find_col_idx(["枠"], [r"Waku"])
                idx_uma   = find_col_idx(["馬番"], [r"Num"])
                idx_name  = find_col_idx(["馬名"], [r"Horse_Info", r"HorseName"])
                idx_sei   = find_col_idx(["性齢"], [r"SexAge"])
                idx_kin   = find_col_idx(["斤量"], [r"Weight", r"Kinryo"])
                idx_kishu = find_col_idx(["騎手"], [r"Jockey"])
                idx_time  = find_col_idx(["タイム"], [r"^Time$"])
                idx_tsuka = find_col_idx(["通過", "通過順"], [r"Pass"])
                idx_agari = find_col_idx(["上り", "後3F", "上がり"], [r"Last3F", r"Agari"])
                idx_tan   = find_col_idx(["単勝"], [r"Odds"])
                idx_ninki = find_col_idx(["人気"], [r"Popular"])
                idx_wt    = find_col_idx(["馬体重", "体重"], [r"Horse_Weight", r"Weight"])
                idx_tr    = find_col_idx(["調教師", "厩舎"], [r"Trainer"])

                rows_found = 0
                for row in table.find_all("tr")[1:]:
                    cols = row.find_all("td")
                    if not cols or len(cols) <= idx_chaku: continue

                    def col_val(idx):
                        return clean_text(cols[idx].text) if idx != -1 and idx < len(cols) else ""

                    raw_chaku = col_val(idx_chaku)
                    m_chaku = re.search(r'\d+', raw_chaku)
                    if not m_chaku: continue

                    h_name = clean_horse_name(col_val(idx_name))

                    raw_wt = col_val(idx_wt)
                    bataiju, diff_wt = "", ""
                    m_wt = re.search(r'(\d{3,4})\s*\(\s*([+-]?\d+)\s*\)', raw_wt)
                    if m_wt:
                        bataiju = m_wt.group(1)
                        diff_wt = m_wt.group(2)
                    elif raw_wt.isdigit():
                        bataiju = raw_wt
                        diff_wt = "0"

                    raw_trainer = col_val(idx_tr)
                    trainer_clean = re.sub(r'\[.*?\]|［.*?］', '', raw_trainer).strip()

                    default_p_data = {"公式脚質": ""}
                    for i in range(1, 6):
                        for k in ["着順", "馬場", "通過順", "上がり3F", "タイム", "馬体重", "体重増減"]:
                            default_p_data[f"前走{i}_{k}"] = ""
                    p_data = past_dict.get(h_name, default_p_data)

                    record = {
                        "date": date_str_hyph,
                        "race_id": str(race_id),
                        "place_name": place_name,
                        "surface": surface,
                        "distance": distance,
                        "track_condition": track_condition,
                        "枠番": col_val(idx_waku),
                        "馬番": col_val(idx_uma),
                        "馬名": h_name,
                        "性齢": col_val(idx_sei),
                        "斤量": col_val(idx_kin),
                        "騎手": col_val(idx_kishu),
                        "調教師": trainer_clean,
                        "人気": col_val(idx_ninki),
                        "単勝": col_val(idx_tan),
                        "馬体重": bataiju,
                        "体重増減": diff_wt,
                        "着順": m_chaku.group(0),
                        "タイム": col_val(idx_time),
                        "通過": col_val(idx_tsuka),
                        "上り": col_val(idx_agari),
                        "公式脚質": p_data.get("公式脚質", "")
                    }

                    for i in range(1, 6):
                        for k in ["着順", "馬場", "通過順", "上がり3F", "タイム", "馬体重", "体重増減"]:
                            record[f"前走{i}_{k}"] = p_data.get(f"前走{i}_{k}", "")

                    daily_data.append(record)
                    rows_found += 1

                if rows_found == 0:
                    break

            except Exception:
                continue

            time.sleep(random.uniform(0.3, 0.5))

    return daily_data

# ==========================================
# メイン処理（勝手な全削除なし・完全レジューム対応）
# ==========================================
if __name__ == "__main__":
    JST = timezone(timedelta(hours=+9), 'JST')
    now = datetime.now(JST)
    default_start_date = now - timedelta(days=365 * 3)  # 直近3年

    print("==================================================")
    print("🚀 nar_keiba_database.csv 安全更新パイプライン")
    print(f"現在日時: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")

    start_date = get_start_date_and_status(DB_FILE, default_start_date)

    if start_date.date() >= now.date():
        print("✅ データベースは既に最新です。追加取得はありません。")
        exit()

    session = create_retry_session()
    current_date = start_date
    saved_days = 0
    total_added = 0

    while current_date.date() < now.date():
        date_str_hyph = current_date.strftime("%Y-%m-%d")

        daily_records = scrape_full_daily_results(session, current_date)

        if daily_records:
            df_day = pd.DataFrame(daily_records)
            write_header = not os.path.exists(DB_FILE) or os.path.getsize(DB_FILE) == 0
            df_day.to_csv(DB_FILE, mode='a', index=False, header=write_header, encoding='utf-8-sig')

            saved_days += 1
            total_added += len(df_day)
            print(f"💾 [{date_str_hyph}] {len(df_day)} 件追記完了 (今回追加累計: {total_added} 件)")
        else:
            print(f"ℹ️ [{date_str_hyph}] 開催なし")

        current_date += timedelta(days=1)
        time.sleep(random.uniform(0.4, 0.7))

    print(f"\n🎉 取得完了！ 今回は合計 {total_added} 件のレース結果を追記しました。")