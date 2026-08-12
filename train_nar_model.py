import os
import re
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# 地方競馬場コードマップ
PLACE_MAP = {
    '30': '門別', '35': '盛岡', '36': '水沢', '42': '浦和', '43': '船橋',
    '44': '大井', '45': '川崎', '46': '金沢', '47': '笠松', '48': '名古屋',
    '50': '園田', '51': '姫路', '54': '高知', '55': '佐賀', '65': '帯広'
}

def parse_time_to_seconds(val):
    """タイム文字列（1:46.6 等）を秒数（106.6）に変換"""
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    try:
        parts = val.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except Exception:
        return np.nan
    return np.nan

def clean_and_build_features(df_raw: pd.DataFrame, output_ml_csv="ml_target_data_chiho.csv") -> pd.DataFrame:
    """生データから自作5大指数およびモデル用特徴量を生成・保存"""
    print("🧹 1. データクリーニングと基本属性の抽出中...")
    df = df_raw.copy()

    # 1. レースIDから日付・開催場を抽出
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['race_id'].str[:8], format='%Y%m%d', errors='coerce')
    df['place_code'] = df['race_id'].str[4:6]
    df['place_name'] = df['place_code'].map(PLACE_MAP).fillna('その他')

    # 2. 着順と目的変数の作成（学習ターゲット）
    df['着順_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df.dropna(subset=['着順_num']).copy()
    df['着順_num'] = df['着順_num'].astype(int)
    df['target_win'] = (df['着順_num'] == 1).astype(int)
    df['target_place'] = (df['着順_num'] <= 3).astype(int)

    # 3. 基本数値項目の変換
    df['time_sec'] = df['タイム'].apply(parse_time_to_seconds)
    df['last_3f'] = pd.to_numeric(df['上り'], errors='coerce')
    df['distance_num'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1400.0)
    df['weight_num'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(54.0)
    df['gate_num'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(8.0)
    
    # 馬体重と増減
    df['horse_weight'] = df['馬体重'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    df['weight_change'] = df['馬体重'].astype(str).str.extract(r'\(([-+]?\d+)\)')[0].astype(float).fillna(0.0)

    # 距離短縮・延長の計算（ヒモ荒れファクター）
    df = df.sort_values(by=['馬名', 'date'])
    df['prev_dist'] = df.groupby('馬名')['distance_num'].shift(1)
    df['dist_change'] = (df['distance_num'] - df['prev_dist']).fillna(0.0)

    # 4. 展開データ：通過順位（1角・4角位置）
    def parse_passing_pos(val):
        if pd.isna(val) or not isinstance(val, str):
            return np.nan, np.nan
        parts = re.findall(r'\d+', str(val))
        if not parts:
            return np.nan, np.nan
        return int(parts[0]), int(parts[-1])

    coords = df['通過'].apply(parse_passing_pos)
    df['first_corner'] = [c[0] for c in coords]
    df['last_corner'] = [c[1] for c in coords]
    
    # 前崩れ（ハイペース）フラグ
    df['is_front'] = (df['first_corner'] <= 3).astype(int)
    df['race_front_count'] = df.groupby('race_id')['is_front'].transform('sum')

    print("💡 2. 地方特化：自作5大指数の生成中...")
    # ① 独自タイム指数
    raw_speed = (df['distance_num'] / df['time_sec']) * 100
    weight_bonus = (df['weight_num'] - 54.0) * 0.5
    df['custom_time_index'] = (raw_speed + weight_bonus).round(1)

    # ② 独自タイム指数M
    race_mean_speed = df.groupby('race_id')['custom_time_index'].transform('mean')
    df['custom_time_index_m'] = (df['custom_time_index'] - race_mean_speed + 80.0).round(1)

    # ③ スタート指数
    df['custom_start_index'] = ((20.0 - df['first_corner'].fillna(10.0)) * 3.5 + df['gate_num'] * 0.4).round(1)

    # ④ 追走指数（修正箇所：df['mid_speed'] をカラムとして明示的に保持）
    mid_dist = df['distance_num'] - 600.0
    mid_time = df['time_sec'] - df['last_3f']
    df['mid_speed'] = (mid_dist / mid_time.clip(lower=1.0)) * 100.0
    race_mid_mean = df.groupby('race_id')['mid_speed'].transform('mean')
    df['custom_pursuit_index'] = ((df['mid_speed'] - race_mid_mean) * 4.0 + 80.0).round(1)

    # ⑤ 上がり指数
    race_last3f_mean = df.groupby('race_id')['last_3f'].transform('mean')
    df['custom_last3f_index'] = ((race_last3f_mean - df['last_3f']) * 8.0 + 80.0).round(1)

    # 騎手データ
    jockey_win = df.groupby('騎手')['target_win'].agg(['count', 'mean']).reset_index()
    jockey_win.columns = ['騎手', 'jockey_runs', 'jockey_win_rate']
    df = df.merge(jockey_win, on='騎手', how='left')

    # 保存
    df = df.sort_values(by=['race_id', '馬番'])
    df.to_csv(output_ml_csv, index=False, encoding='utf-8-sig')
    print(f"📁 加工済みデータセットを保存しました: {output_ml_csv} (行数: {len(df)}件)")
    return df

def train_nar_lightgbm(df: pd.DataFrame, model_path="keiba_ai_model_nar.pkl"):
    """地方競馬用LightGBMモデルの学習"""
    print("\n🚀 3. LightGBMモデルの学習を開始します...")
    
    features = [
        'custom_time_index', 'custom_time_index_m',
        'custom_start_index', 'custom_pursuit_index', 'custom_last3f_index',
        'distance_num', 'dist_change',
        'weight_num', 'gate_num',
        'first_corner', 'is_front', 'race_front_count',
        'horse_weight', 'weight_change',
        'jockey_win_rate', 'jockey_runs'
    ]

    X = df[features].fillna(0.0)
    y = df['target_win']

    races = df['race_id'].unique()
    split_idx = int(len(races) * 0.8)
    train_races = set(races[:split_idx])
    test_races = set(races[split_idx:])

    train_mask = df['race_id'].isin(train_races)
    test_mask = df['race_id'].isin(test_races)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=800,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, preds)
    print(f"\n🏆 検証データ AUCスコア: {auc:.4f} (0.5以上ならAIに予測能力あり)")

    model_payload = {
        'model': model,
        'features': features
    }
    joblib.dump(model_payload, model_path)
    print(f"📦 学習済みモデルを保存しました: {model_path}")

    imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    print("\n📊 特徴量重要度（どのデータが予測に一番効いているか）:")
    print(imp.to_string(index=False))

if __name__ == "__main__":
    raw_csv = "nar_keiba_database.csv"
    if os.path.exists(raw_csv):
        df_raw = pd.read_csv(raw_csv, dtype={'race_id': str})
        df_ml = clean_and_build_features(df_raw)
        train_nar_lightgbm(df_ml)
    else:
        print(f"🚨 {raw_csv} が見つかりません。")