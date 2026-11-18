import os
import sys
import pandas as pd
from feature_engineering import generate_features

INPUT_CSV = "nar_keiba_database.csv"
OUTPUT_CSV = "ml_target_data_chiho.csv"

def main():
    print("\n" + "="*50)
    print(" 🚀 地方競馬AI 勝ち子ちゃん 【学習データ生成（共通モジュール完全連動）】")
    print("="*50)

    if not os.path.exists(INPUT_CSV):
        print(f"❌ 入力ファイル '{INPUT_CSV}' が見つかりません。")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig', dtype={'race_id': str})
    print(f"  [INFO] 元データ読み込み: {len(df):,} 行")

    df['race_id_str'] = df['race_id'].astype(str)
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['date_parsed', 'race_id_str']).reset_index(drop=True)

    clean_rank = pd.to_numeric(df['着順'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['target_win'] = (clean_rank == 1).astype(int)
    df['target_place'] = (clean_rank <= 3).astype(int)
    df['relevance'] = clean_rank.map({1: 5, 2: 3, 3: 2, 4: 1, 5: 1}).fillna(0).astype(int)

    print("  [INFO] 特徴量生成エンジンを実行中...")
    df = generate_features(df, is_train=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("="*50)
    print(f"🎉 学習データの生成が完了しました！ ({len(df):,} 行 / {df.shape[1]} 列)")
    print(f"📁 保存先: {OUTPUT_CSV}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()