import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import os

print("🤖 勝ち子ちゃん AI機械学習スクリプト (LightGBM)")
CSV_FILE = "ml_target_data_chiho.csv"
MODEL_FILE = "keiba_ai_model_nar.pkl"

if not os.path.exists(CSV_FILE):
    print("⚠️ 過去データがありません。ダミーデータを生成して初期モデルを作ります...")
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'race_id': np.random.randint(1000, 9999, n),
        '馬名': ['Horse'+str(i) for i in range(n)],
        '単勝': np.random.uniform(1.5, 150.0, n),
        '人気': np.random.randint(1, 14, n),
        '斤量': np.random.uniform(50.0, 58.0, n),
        '馬番': np.random.randint(1, 14, n),
        'first_corner': np.random.uniform(1.0, 14.0, n),
        'target_rank': np.random.randint(1, 14, n),
    })
    df['target_win'] = (df['target_rank'] == 1).astype(int)
    df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
else:
    print("📊 過去のレースデータを読み込みます...")
    df = pd.read_csv(CSV_FILE)

# 学習に使う要素
features = ['単勝', '人気', '斤量', '馬番', 'first_corner']
for f in features:
    if f not in df.columns: df[f] = 0.0
        
X = df[features].fillna(0)
y = df['target_win'] if 'target_win' in df.columns else np.random.randint(0, 2, len(df))

print("🧠 LightGBMモデルの学習を開始...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"✨ 学習完了！ テスト精度: {score:.3f}")

joblib.dump({'model': model, 'features': features}, MODEL_FILE)
print(f"💾 モデルを保存しました: {MODEL_FILE}")
print("これで勝ち子ちゃんのAI予想精度がアップデートされました！")