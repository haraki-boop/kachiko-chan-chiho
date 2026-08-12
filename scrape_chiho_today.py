import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

NAR_PLACES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

def clean_text(text):
    """余分な改行やスペースを完全に除去"""
    if not text:
        return ""
    return re.sub(r'[\s\u3000]+', '', str(text)).strip()

def smart_decode(content):
    """netkeibaのUTF-8化と旧EUC-JPの両方に自動対応して文字化けを完全防止"""
    try:
        # まずは現代の主流であるUTF-8でデコード
        return content.decode('utf-8')
    except UnicodeDecodeError:
        # 失敗した場合は従来のEUC-JPでデコード
        return content.decode('euc-jp', errors='replace')

def scrape_race_card(race_id, date_str, is_test=False):
    """レースIDから出馬表を取得"""
    url = f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.7,en;q=0.3"
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        
        # 🚨ここが文字化け防止の要
        html = smart_decode(res.content)
        soup = BeautifulSoup(html, "html.parser")
        
        table = soup.find("table", class_=re.compile("Shutuba_Table|RaceTable", re.I))
        if not table:
            for t in soup.find_all("table"):
                if "馬番" in t.get_text() and "馬名" in t.get_text():
                    table = t
                    break
                    
        if not table:
            return None
            
        race_name = "地方一般"
        distance = "1400"
        
        r_name_div = soup.find("div", class_=re.compile("RaceName", re.I))
        if r_name_div:
            race_name = clean_text(r_name_div.text)
            
        r_data_div = soup.find("div", class_=re.compile("RaceData", re.I))
        if r_data_div:
            dist_match = re.search(r'(\d{3,4})m', r_data_div.text)
            if dist_match:
                distance = dist_match.group(1)
        
        rows = []
        # ヘッダー行を飛ばしてデータ行だけを抽出
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 10:
                continue
                
            try:
                # 1. 馬番（2番目の列: index 1）
                umaban = clean_text(tds[1].text)
                if not umaban.isdigit(): continue
                
                # 2. 馬名（4番目の列: index 3）
                horse_name = clean_text(tds[3].text)
                if not horse_name: continue
                
                # 3. 斤量（6番目の列: index 5）
                weight = 54.0
                w_str = clean_text(tds[5].text)
                w_match = re.search(r'([0-9\.]+)', w_str)
                if w_match:
                    weight = float(w_match.group(1))
                        
                # 4. 騎手名（7番目の列: index 6）
                jockey_name = clean_text(tds[6].text)
                
                # 5. 単勝オッズ（10番目の列: index 9）
                odds = 15.0
                odds_str = clean_text(tds[9].text)
                if "---" not in odds_str and odds_str != "":
                    odds_match = re.search(r'([0-9\.]+)', odds_str)
                    if odds_match:
                        odds = float(odds_match.group(1))
                    
                # 6. 人気（11番目の列: index 10）
                pop = 99
                pop_str = clean_text(tds[10].text)
                pop_match = re.search(r'([0-9]+)', pop_str)
                if pop_match:
                    pop = int(pop_match.group(1))
                        
                rows.append({
                    "date": date_str,
                    "race_id": race_id,
                    "race_name": race_name,
                    "distance": distance,
                    "馬番": umaban,
                    "馬名": horse_name,
                    "斤量": weight,
                    "騎手": jockey_name,
                    "単勝": odds,
                    "人気": pop
                })
            except Exception:
                continue
                
        return pd.DataFrame(rows)
    except Exception as e:
        if is_test:
            print(f"  [エラー] {e}")
        return None

def collect_todays_races():
    JST = timezone(timedelta(hours=+9), 'JST')
    today = datetime.now(JST)
    year = today.strftime("%Y")
    mmdd = today.strftime("%m%d")
    date_formatted = today.strftime("%Y-%m-%d")
    
    print(f"📅 取得対象日: {date_formatted} の地方競馬データを取得します...")
    
    all_dfs = []
    active_places = []
    
    print("🔍 本日の開催競馬場を調査中...")
    for code, name in NAR_PLACES.items():
        for r_test in ["01", "02", "03"]:
            test_race_id = f"{year}{code}{mmdd}{r_test}"
            df = scrape_race_card(test_race_id, date_formatted, is_test=True)
            if df is not None and not df.empty:
                active_places.append(code)
                print(f"  🎯 {name}競馬場の開催を確認！ (検知: {r_test}R)")
                break
        time.sleep(0.4)
        
    if not active_places:
        print("⚠️ どの競馬場も開催されていないか、取得できませんでした。")
        return

    print(f"\n🔍 開催競馬場の全レースを取得中...")
    for code in active_places:
        place_name = NAR_PLACES[code]
        empty_count = 0
        for r_num in range(1, 13):
            race_id = f"{year}{code}{mmdd}{r_num:02d}"
            df = scrape_race_card(race_id, date_formatted)
            
            if df is not None and not df.empty:
                all_dfs.append(df)
                print(f"  [成功] {place_name} {r_num}R ({len(df)}頭)")
                empty_count = 0
            else:
                empty_count += 1
                if empty_count >= 2:
                    break
            time.sleep(0.6)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("future_races_chiho.csv", index=False, encoding='utf-8-sig')
        print(f"\n✨ 取得完了！ 'future_races_chiho.csv' を保存しました。")
    else:
        print("取得できる出馬表データがありませんでした。")

if __name__ == "__main__":
    collect_todays_races()