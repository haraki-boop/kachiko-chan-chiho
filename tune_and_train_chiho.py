import pandas as pd
import numpy as np
import re
import os
import optuna
import joblib
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

print("==================================================")
print("🌸 勝ち子ちゃん 脳みそアップデート＆自動チューニング (的中率特化)")
print("==================================================")

CSV_FILE = "ml_target_data_chiho.csv"

if not os.path.exists(CSV_FILE):
    print("⚠️ データが見つかりません。")
    exit()

print("📊 過去データを読み込み中...")
df = pd.read_csv(CSV_FILE, low_memory=False)

# --------------------------------------------------------
# 🎯 1. ベース特徴量の処理
# --------------------------------------------------------
def parse_rank(x):
    if pd.isna(x): return 6.0
    s = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s)
    except: return 6.0

df['target_rank'] = df.get('着順', df.get('着順_num')).apply(parse_rank)
df = df[df['target_rank'] < 90.0].copy()

df['first_corner'] = pd.to_numeric(df.get('first_corner', df.get('1角')), errors='coerce').fillna(8.0)
df['last_corner'] = pd.to_numeric(df.get('last_corner', df.get('4角')), errors='coerce').fillna(df['first_corner'])
df['corner_diff'] = df['first_corner'] - df['last_corner']
df['last_3f'] = pd.to_numeric(df.get('last_3f', df.get('上り')), errors='coerce').fillna(39.0)
df['time_diff'] = pd.to_numeric(df.get('time_diff', df.get('着差')), errors='coerce').fillna(1.5)
df['斤量'] = pd.to_numeric(df.get('斤量'), errors='coerce').fillna(54.0)
df['馬番_num'] = pd.to_numeric(df.get('馬番'), errors='coerce').fillna(1.0)
df['waku_num'] = pd.to_numeric(df.get('枠番'), errors='coerce').fillna(1.0)
df['distance_num'] = pd.to_numeric(df.get('distance'), errors='coerce').fillna(1400.0)
df['馬名_clean'] = df['馬名'].astype(str).apply(lambda x: re.sub(r'[\s\u3000]+', '', str(x)))
df['place_code'] = df['race_id'].astype(str).str[4:6].astype(float)
df['time_sec'] = pd.to_numeric(df.get('time_sec', df.get('タイム')), errors='coerce')

df['recent_avg_rank_3'] = df.groupby('馬名_clean')['target_rank'].transform(lambda x: x.shift().rolling(3, min_periods=1).mean()).fillna(6.0)

# --------------------------------------------------------
# 🔥 2. 【新規追加】最強の3大特徴量の生成
# --------------------------------------------------------
print("✨ 最強の追加特徴量（スピード偏差・枠脚質バイアス・賞金変動）を生成中...")

# ① 持ちタイム（スピード絶対値）
# 距離 / タイム = 秒速 を計算し、馬ごとに過去の最高スピードを算出
df['speed_m_per_sec'] = df['distance_num'] / df['time_sec'].clip(lower=10.0)
df['best_speed'] = df.groupby('馬名_clean')['speed_m_per_sec'].transform(lambda x: x.shift().rolling(10, min_periods=1).max()).fillna(15.0)

# ② 枠順×脚質の有利不利インデックス
# 1角の通過順位を脚質（1=逃げ, 2=先行, 3=差し追込）に分類
df['kyakushitsu_cat'] = pd.cut(df['first_corner'], bins=[0, 2.5, 5.5, 99], labels=[1, 2, 3]).astype(float)
df['prev_kyakushitsu'] = df.groupby('馬名_clean')['kyakushitsu_cat'].shift().fillna(2.0)
# コース×距離×枠番×前走脚質 ごとの「3着内率」を事前計算してマッピング
df['target_place_flag'] = (df['target_rank'] <= 3.0).astype(int)
bias_map = df.groupby(['place_code', 'distance_num', 'waku_num', 'prev_kyakushitsu'])['target_place_flag'].transform('mean').fillna(0.2)
df['course_bias_index'] = bias_map

# ③ 実質クラス変動（賞金の増減）
# 前走の獲得賞金との差分で、相手関係の強化（マイナス評価）か弱化（プラス評価）を数値化
df['prize_money'] = pd.to_numeric(df.get('賞金(万円)'), errors='coerce').fillna(0.0)
df['prev_prize'] = df.groupby('馬名_clean')['prize_money'].shift().fillna(0.0)
df['prize_diff'] = df['prev_prize'] - df['prize_money'] # プラスなら今回が格下戦（激アツ）

features = [
    'first_corner', 'last_corner', 'corner_diff', 'last_3f', 'time_diff',
    '斤量', '馬番_num', 'waku_num', 'distance_num', 'place_code',
    'recent_avg_rank_3', 
    'best_speed', 'course_bias_index', 'prize_diff' # 👈 追加された3つの新・最強特徴量
]

# --------------------------------------------------------
# 🧠 3. 学習データと検証データに分割（時系列）
# --------------------------------------------------------
split_idx = int(len(df) * 0.80)
train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()

X_train = train_df[features].fillna(0.0)
y_train_place = (train_df['target_rank'] <= 3.0).astype(int) # 的中率(3着内)に極振り

X_val = val_df[features].fillna(0.0)
y_val_place = (val_df['target_rank'] <= 3.0).astype(int)

# --------------------------------------------------------
# 🤖 4. Optunaによる自動最適化チューニング
# --------------------------------------------------------
print("⚙️ AIが『手堅さ（3着内率）』に特化した黄金パラメータを探索中... (数分かかります)")

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_iter': trial.suggest_int('max_iter', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
        'random_state': 42
    }
    
    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train_place)
    
    # 評価指標：AUC（どれだけ正確に「強い馬」を上位にランク付けできたか）
    preds = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val_place, preds)
    return auc_score

# 30回テストして最高の組み合わせを探す（数字を増やせばさらに賢くなりますが時間はかかります）
study = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING) # ログをスッキリさせる
study.optimize(objective, n_trials=30)

print("\n🏆 チューニング完了！")
print(f"🥇 最高スコア (AUC): {study.best_value:.4f}")
print("🥇 黄金パラメータ:", study.best_params)

# --------------------------------------------------------
# 💾 5. 最高の脳みそで本番学習＆保存
# --------------------------------------------------------
print("\n💾 黄金パラメータを使って最終モデルを学習中...")

best_params = study.best_params
best_params['random_state'] = 42

# 3着内（複勝圏）特化モデル
model_place = HistGradientBoostingClassifier(**best_params)
model_place.fit(X_train, y_train_place)

# 1着（単勝）モデルも同じパラメータで学習
y_train_win = (train_df['target_rank'] == 1.0).astype(int)
model_win = HistGradientBoostingClassifier(**best_params)
model_win.fit(X_train, y_train_win)

# 保存用データのパッキング
model_data = {
    'features': features,
    'model_place': model_place,
    'model_win': model_win,
    'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

MODEL_FILE = "keiba_ai_model_nar.pkl"
joblib.dump(model_data, MODEL_FILE)
print(f"🎉 脳みそのアップデートが完了しました！ ({MODEL_FILE} に保存済)")
print("👉 これでWebアプリ（勝ち子ちゃん）の予測精度が飛躍的に向上します！")