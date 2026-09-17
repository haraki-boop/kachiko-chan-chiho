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

def get_race_bbs(race_id):
    url = f"https://nar.netkeiba.com/race/bbs.html?race_id={race_id}"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        comments = []
        for item in soup.find_all(["div", "p", "span"], class_=re.compile(r'Comment_Text|Bbs_Text|txt|comment', re.I)):
            c = clean_text(item.text)
            if c and len(c) >= 5 and "※" not in c and "コメントを投稿" not in c and "取得不可" not in c:
                comments.append(c)
            if len(comments) >= 3:
                break
        
        if not comments: return "特になし"
        return " / ".join(comments)
    except Exception:
        return "取得不可"

def get_active_venues(date_str):
    url = f"https://nar.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    print(f"🔍 本日の開催会場をサーチ中... ({url})")
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(res.content, 'html.parser')
        active_places = set()
        
        year = date_str[:4]
        mmdd = date_str[4:]
        pattern = rf'race_id={year}(\d{{2}}){mmdd}'
        
        for a in soup.find_all('a', href=re.compile(pattern)):
            match = re.search(pattern, a['href'])
            if match:
                active_places.add(match.group(1))
        
        if active_places:
            print(f"🎯 本日の開催会場を発見: {[NAR_PLACES.get(p, p) for p in active_places]}")
            return list(active_places)
        else:
            return list(NAR_PLACES.keys())
    except Exception:
        return list(NAR_PLACES.keys())

def get_today_chiho_races():
    JST = timezone(timedelta(hours=+9), 'JST')
    today_dt = datetime.now(JST)
    
    date_str_hyph = today_dt.strftime("%Y-%m-%d")
    date_str_raw = today_dt.strftime("%Y%m%d")
    year = today_dt.strftime("%Y")
    mmdd = today_dt.strftime("%m%d")
    
    print(f"🌸 今日の地方競馬データを取得中... ({date_str_hyph})")
    active_places = get_active_venues(date_str_raw)
    headers = {"User-Agent": "Mozilla/5.0"}
    all_races = []

    for place_code in active_places:
        place_name = NAR_PLACES.get(place_code, "地方")
        print(f"📍 {place_name}競馬場の開催データを取得中...")
        
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

                race_name_tag = soup.find("div", class_="RaceName")
                race_name = clean_text(race_name_tag.text) if race_name_tag else f"{r_num}R"
                
                data_intro = soup.find("div", class_="RaceData01") or soup.find("div", class_="RaceData00")
                distance = 1400
                if data_intro:
                    dist_match = re.search(r'(\d{3,4})m', data_intro.text)
                    if dist_match: distance = int(dist_match.group(1))

                print(f"   💬 {r_num}R の出馬表と世論を取得中...")
                bbs_comment = get_race_bbs(race_id)
                time.sleep(0.5)

                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all("td")
                    # netkeibaの出馬表は通常8〜10列以上ある。列数が足りない行（ヘッダ等）はスキップ
                    if len(cols) < 8: continue
                    
                    # 🚨 クラス名ではなく、表の「列インデックス」で確実に取得する（仕様変更に強い）
                    wakuban = clean_text(cols[0].text)
                    umaban = clean_text(cols[1].text)
                    
                    if not umaban.isdigit(): continue 
                    
                    # 馬名セル（馬のプロフィールリンクやアイコンが含まれる）
                    horse_name = clean_text(cols[3].text)
                    
                    # 性齢、斤量、騎手、調教師、馬体重（基本レイアウトに準拠）
                    sei_rei = clean_text(cols[4].text)
                    kinryo = clean_text(cols[5].text)
                    jockey = clean_text(cols[6].text)
                    trainer = clean_text(cols[7].text)
                    weight = clean_text(cols[8].text) if len(cols) > 8 else "470(0)"
                    
                    # オッズと人気は後ろの列にある。無い場合はデフォルト値
                    odds = clean_text(cols[9].text) if len(cols) > 9 else "15.0"
                    pop = clean_text(cols[10].text) if len(cols) > 10 else "99"

                    # 取り出したデータが空の場合の安全策
                    if not horse_name: continue # 馬名が取れなかったら異常行としてスキップ
                    if odds in ["---", "", "0.0", "＊＊", "**"]: odds = "15.0"
                    if pop in ["---", "", "0", "＊＊", "**"]: pop = "99"
                    if weight in ["---", ""]: weight = "470(0)"
                    if not kinryo: kinryo = "54.0"

                    all_races.append({
                        "date": date_str_hyph, "race_id": str(race_id), "place_name": place_name,
                        "r_num": r_num, "race_name": race_name, "distance": distance,
                        "枠番": wakuban, "馬番": umaban, "馬名": horse_name, "性齢": sei_rei,
                        "斤量": kinryo, "騎手": jockey, "調教師": trainer, "馬体重": weight,
                        "オッズ": odds, "人気": pop,
                        "世論コメント": bbs_comment
                    })
                time.sleep(0.2)
            except Exception as e:
                print(f"   ⚠️ {r_num}R でエラー: {e}")
                continue

    return pd.DataFrame(all_races)

if __name__ == "__main__":
    df = get_today_chiho_races()
    if not df.empty:
        # UTF-8 BOM付きで保存（ExcelやPandasでの文字化けを防止）
        df.to_csv("future_races_chiho.csv", index=False, encoding='utf-8-sig')
        print(f"✨ 成功: {len(df)} 件の正常な出馬表データを保存しました！")
    else:
        print("🚨 エラー: 本日の出馬表データが1件も取得できませんでした。")