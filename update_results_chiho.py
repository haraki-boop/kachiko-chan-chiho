import os
import re
import time
import subprocess
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone

ML_TARGET_CSV = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

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
                
                umaban = clean_text(tds[2].text)
                horse_name = clean_text(tds[3].text)
                sei_rei = clean_text(tds[4].text)
                weight = float(re.search(r'([0-9\.]+)', clean_text(tds[5].text)).group(1))
                jockey = clean_text(tds[6].text)
                
                horse_weight = clean_text(tds[14].text) if len(tds) > 14 else ""
                
                passage = clean_text(tds[10].text)
                first_corner = float(passage.split('-')[0]) if '-' in passage and passage.split('-')[0].isdigit() else 8.0
                
                odds = float(clean_text(tds[12].text)) if clean_text(tds[12].text).replace('.','',1).isdigit() else 15.0
                pop = int(clean_text(tds[13].text)) if clean_text(tds[13].text).isdigit() else 99
                
                target_win = 1 if rank == 1 else 0
                
                rows.append({
                    "date": date_str, "race_id": race_id, "馬名": horse_name,
                    "性齢": sei_rei, "単勝": odds, "人気": pop, "斤量": weight,
                    "馬番": umaban, "騎手": jockey, "馬体重": horse_weight,
                    "first_corner": first_corner, "target_rank": rank, "target_win": target_win
                })
            except Exception:
                continue
                
        return pd.DataFrame(rows)
    except Exception:
        return None

def main():
    JST = timezone(timedelta(hours=+9), 'JST')
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
            combined = pd.concat([old_data, new_data]).drop_duplicates(subset=['race_id', '馬番'], keep='last')
        else:
            combined = new_data
            
        combined.to_csv(ML_TARGET_CSV, index=False, encoding='utf-8-sig')
        print(f"💾 {ML_TARGET_CSV} を更新しました！")
        
        print("\n🧠 AIモデルを自動再学習中...")
        subprocess.run(["python", "train_lgbm_model.py"], check=True)
        
        print("\n🚀 クラウドへ同期中...")
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"update: {date_formatted} results & retrain AI"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🎉 すべての更新が完了しました！")
    else:
        print("⚠️ 取得できるレース結果がありませんでした。")

if __name__ == "__main__":
    main()