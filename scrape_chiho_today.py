import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime, timedelta, timezone
import os

NAR_PLACES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

def clean_text(text):
    if not text: return ""
    return re.sub(r'[\s\u3000]+', '', str(text)).strip()

def get_today_chiho_races():
    JST = timezone(timedelta(hours=+9), 'JST')
    # 🔥 修正①: 実行日の「1日前（昨日）」のデータを取得するように変更
    target_dt = datetime.now(JST) - timedelta(days=1)
    
    date_str = target_dt.strftime("%Y-%m-%d")
    year = target_dt.strftime("%Y")
    mmdd = target_dt.strftime("%m%d")
    
    print(f"🌸 指定日（昨日）の地方競馬データを取得中... ({date_str})")
    headers = {"User-Agent": "Mozilla/5.0"}
    all_races = []

    for place_code, place_name in NAR_PLACES.items():
        place_found = False
        for r_num in range(1, 13):
            race_id = f"{year}{place_code}{mmdd}{r_num:02d}"
            url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
            
            try:
                res = requests.get(url, headers=headers, timeout=8)
                if res.status_code != 200:
                    if r_num == 1: break
                    else: continue

                soup = BeautifulSoup(res.content, 'html.parser')
                table = soup.find("table", class_=re.compile("RaceTable01|Shutuba_Table", re.I))
                if not table:
                    if r_num == 1: break
                    else: continue
                    
                if not place_found:
                    print(f"📍 {place_name}競馬場の開催データを取得中...")
                    place_found = True

                race_name_tag = soup.find("div", class_="RaceName")
                race_name = clean_text(race_name_tag.text) if race_name_tag else f"{r_num}R"
                
                data_intro = soup.find("div", class_="RaceData01") or soup.find("div", class_="RaceData00")
                distance = 1400
                if data_intro:
                    dist_match = re.search(r'(\d{3,4})m', data_intro.text)
                    if dist_match: distance = int(dist_match.group(1))

                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) < 7: continue
                    
                    wakuban = clean_text(cols[0].text)
                    umaban = clean_text(cols[1].text)
                    if not umaban.isdigit(): continue 
                    
                    horse_name = clean_text(cols[3].text)
                    sei_rei = clean_text(cols[4].text)
                    kinryo = clean_text(cols[5].text)
                    jockey = clean_text(cols[6].text)
                    
                    # 🔥 修正②: 不安定な「馬体重」のスクレイピング処理を完全に削除しました
                    
                    odds_td = row.find("td", id=re.compile(r'odds-', re.I))
                    odds = clean_text(odds_td.text) if odds_td else "15.0"
                    if odds in ["---", "", "0.0"]: odds = "15.0"
                    
                    pop_td = row.find("td", id=re.compile(r'pop-', re.I))
                    pop = clean_text(pop_td.text) if pop_td else "99"

                    all_races.append({
                        "date": date_str, "race_id": str(race_id), "place_name": place_name,
                        "r_num": r_num, "race_name": race_name, "distance": distance,
                        "枠番": wakuban, "馬番": umaban, "馬名": horse_name, "性齢": sei_rei,
                        "斤量": kinryo, "騎手": jockey,
                        "単勝": odds, "人気": pop
                    })
                time.sleep(0.2)
            except Exception:
                continue

    return pd.DataFrame(all_races)

if __name__ == "__main__":
    df = get_today_chiho_races()
    if not df.empty:
        df.to_csv("future_races_chiho.csv", index=False, encoding='utf-8-sig')
        print(f"✨ 成功: {len(df)} 件のデータを保存しました！（馬体重は除外しています）")