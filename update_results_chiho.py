import os
import re
import time
import subprocess
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import sys

ML_TARGET_CSV = "ml_target_data_chiho.csv"

NAR_PLACES = {
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀", "65": "帯広"
}

def clean_text(text):
    return re.sub(r'[\s\u3000]+', '', str(text)).strip() if text else ""

def scrape_race_result(race_id, date_str):
    url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'euc-jp'
        soup = BeautifulSoup(res.text, "html.parser")
        
        race_data_div = soup.find("div", class_=re.compile("RaceData01", re.I))
        distance = 1400
        baba = "良"
        if race_data_div:
            rd_text = clean_text(race_data_div.text)
            dist_m = re.search(r'([0-9]{4})m', rd_text)
            if dist_m: distance = int(dist_m.group(1))
            baba_m = re.search(r'馬場:([^\s]+)', race_data_div.text)
            if baba_m: baba = clean_text(baba_m.group(1))

        table = soup.find("table", class_=re.compile("RaceTable01", re.I))
        if not table: return None
            
        rows = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 10: continue
                
            try:
                rank_str = clean_text(tds[0].text)
                if not rank_str.isdigit(): continue
                rank = int(rank_str)
                
                waku = clean_text(tds[1].text) if len(tds) > 1 else ""
                umaban = clean_text(tds[2].text)
                horse_name = clean_text(tds[3].text)
                sei_rei = clean_text(tds[4].text)
                weight = float(re.search(r'([0-9\.]+)', clean_text(tds[5].text)).group(1))
                jockey = clean_text(tds[6].text)
                
                time_diff_raw = clean_text(tds[8].text) if len(tds) > 8 else ""
                time_diff = float(time_diff_raw) if time_diff_raw.replace('.','',1).isdigit() else (0.0 if rank == 1 else 1.5)
                
                passage = clean_text(tds[10].text)
                first_corner = float(passage.split('-')[0]) if '-' in passage and passage.split('-')[0].isdigit() else 8.0
                
                last_3f_raw = clean_text(tds[11].text) if len(tds) > 11 else ""
                last_3f = float(last_3f_raw) if last_3f_raw.replace('.','',1).isdigit() else 39.0

                odds = float(clean_text(tds[12].text)) if clean_text(tds[12].text).replace('.','',1).isdigit() else 15.0
                pop = int(clean_text(tds[13].text)) if clean_text(tds[13].text).isdigit() else 99
                horse_weight = clean_text(tds[14].text) if len(tds) > 14 else ""
                trainer = clean_text(tds[18].text) if len(tds) > 18 else ""
                
                # 「着順」列を過去データと揃えるための修正
                rows.append({
                    "date": date_str, "race_id": race_id, "馬名": horse_name,
                    "性齢": sei_rei, "単勝": odds, "人気": pop, "斤量": weight,
                    "枠番": waku, "馬番": umaban, "騎手": jockey, "調教師": trainer, 
                    "馬体重": horse_weight, "first_corner": first_corner,
                    "last_3f": last_3f, "time_diff": time_diff, 
                    "distance": distance, "馬場": baba,
                    "着順": rank  # ← ここを修正
                })
            except Exception:
                continue
                
        return pd.DataFrame(rows)
    except Exception:
        return None

def main():
    # コマンドライン引数で日付指定があればそれを優先 (--date 2026-08-25)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='Date in YYYY-MM-DD')
    args = parser.parse_args()

    JST = timezone(timedelta(hours=+9), 'JST')
    
    if args.date:
        date_formatted = args.date
        year = date_formatted[0:4]
        mmdd = date_formatted[5:7] + date_formatted[8:10]
    else:
        target_date = datetime.now(JST) - timedelta(days=1)
        print(f"📅 結果を取得したい日付を 8桁 で入力してください（例: 20260819）")
        default_date = target_date.strftime("%Y%m%d")
        user_input = input(f"そのままEnterを押すと昨日({default_date})になります: ").strip()
        
        if len(user_input) == 8 and user_input.isdigit():
            year = user_input[0:4]
            mmdd = user_input[4:8]
            date_formatted = f"{year}-{user_input[4:6]}-{user_input[6:8]}"
        else:
            year = target_date.strftime("%Y")
            mmdd = target_date.strftime("%m%d")
            date_formatted = target_date.strftime("%Y-%m-%d")
    
    print(f"\n📅 対象日: {date_formatted} のレース結果を回収します...")
    all_dfs = []
    
    for code, name in NAR_PLACES.items():
        test_id = f"{year}{code}{mmdd}01"
        if scrape_race_result(test_id, date_formatted) is not None:
            print(f"  🎯 {name}競馬場のデータを回収中...")
            for r_num in range(1, 13):
                race_id = f"{year}{code}{mmdd}{r_num:02d}"
                df = scrape_race_result(race_id, date_formatted)
                if df is not None and not df.empty:
                    all_dfs.append(df)
                time.sleep(0.3)
                
    if all_dfs:
        new_data = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✨ {len(new_data)} 件の最新結果を取得しました！")
        
        if os.path.exists(ML_TARGET_CSV):
            old_data = pd.read_csv(ML_TARGET_CSV, low_memory=False)
            
            # 8/25の古い欠損データを一度削除して綺麗なデータで入れ替える
            old_data = old_data[old_data['date'] != date_formatted]
            
            combined = pd.concat([old_data, new_data]).drop_duplicates(subset=['race_id', '馬番'], keep='last')
        else:
            combined = new_data
            
        combined.to_csv(ML_TARGET_CSV, index=False, encoding='utf-8-sig')
        print(f"💾 {ML_TARGET_CSV} を更新しました！")
        print("🎉 着順の更新が完了しました。続けて python evaluate_yesterday.py を実行してください。")
    else:
        print("⚠️ 取得できるレース結果がありませんでした。")

if __name__ == "__main__":
    main()