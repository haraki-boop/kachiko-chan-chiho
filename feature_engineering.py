import pandas as pd
import numpy as np
import re
import unicodedata
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def clean_name(val):
    if pd.isna(val): return ""
    s = unicodedata.normalize('NFKC', str(val))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def parse_time_to_sec(val):
    if pd.isna(val) or str(val).strip() in ["", "nan", "NaN", "---"]: return np.nan
    s = str(val).strip()
    m = re.match(r'(?:(\d+)[:.])?(\d{1,2})\.(\d+)', s)
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        return mins * 60 + int(m.group(2)) + float('0.' + m.group(3))
    try: return float(s)
    except: return np.nan

def parse_passing_first(val):
    if pd.isna(val) or val == "": return np.nan
    parts = re.findall(r'\d+', str(val))
    return float(parts[0]) if parts else np.nan

def clean_weight_num(val):
    if pd.isna(val): return np.nan
    m = re.search(r'(\d+)', str(val))
    return float(m.group(1)) if m else np.nan

def generate_features(df: pd.DataFrame, is_train: bool = True, past_j_stats=None, past_t_stats=None, past_jp_stats=None) -> pd.DataFrame:
    df = df.copy()

    # 1. 時系列ソートの厳格な強制
    df['race_id_str'] = df['race_id'].astype(str)
    if 'date' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values(['date_parsed', 'race_id_str']).reset_index(drop=True)
    else:
        df = df.sort_values(['race_id_str']).reset_index(drop=True)

    df['place_code_str'] = df['race_id_str'].str[4:6]
    df['馬名_clean'] = df['馬名'].apply(clean_name)
    df['騎手_clean'] = df['騎手'].apply(clean_name)
    df['調教師_clean'] = df['調教師'].apply(clean_name)
    df['j_place_key'] = df['騎手_clean'] + "_" + df['place_code_str']

    # 2. 基礎数値変換
    df['distance_num'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1400.0)
    df['is_turf'] = (df['surface'].astype(str).str.contains('芝')).astype(int)
    df['turn_direction'] = df['place_code_str'].apply(lambda x: 1 if x in ['35', '42', '43', '45'] else 0)
    df['is_heavy_track'] = df.get('track_condition', '良').astype(str).str.contains('重|不良|稍重').astype(int)
    
    df['gate_num'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(8.0)
    df['waku_num'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(4.0)
    df['weight_num'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(54.0)

    if '体重増減' not in df.columns:
        extracted = df['馬体重'].astype(str).str.extract(r'(\d+)\s*\(\s*([+-]?\d+)\s*\)')
        df['body_weight'] = pd.to_numeric(extracted[0], errors='coerce')
        df['weight_change'] = pd.to_numeric(extracted[1], errors='coerce')
        fallback = pd.to_numeric(df['馬体重'].apply(clean_weight_num), errors='coerce')
        df['body_weight'] = df['body_weight'].fillna(fallback).fillna(460.0)
        df['weight_change'] = df['weight_change'].fillna(0.0)
    else:
        df['body_weight'] = pd.to_numeric(df['馬体重'].apply(clean_weight_num), errors='coerce').fillna(460.0)
        df['weight_change'] = pd.to_numeric(df['体重増減'], errors='coerce').fillna(0.0)

    df['kinryo_ratio'] = df['weight_num'] / df['body_weight']
    df['weight_change_ratio'] = df['weight_change'] / df['body_weight']

    df['is_style_nige'] = (df['公式脚質'].astype(str) == '逃').astype(int)
    df['is_style_sen']  = (df['公式脚質'].astype(str) == '先').astype(int)
    df['is_style_sashi']= (df['公式脚質'].astype(str) == '差').astype(int)
    df['is_style_oikomi']=(df['公式脚質'].astype(str) == '追').astype(int)

    # 💡 JRA版移植：レース内の先行激化度（逃げ馬頭数＆ハイペース差し有利フラグ）
    df['race_nige_count'] = df.groupby('race_id_str')['is_style_nige'].transform('sum')
    df['pace_high_advantage'] = ((df['race_nige_count'] >= 3) & (df['公式脚質'].astype(str).isin(['差', '追']))).astype(int)

    # 3. 距離別基準上がり補正付き スピード指数の算出
    # 短距離(<=1200)は基準37.5s、マイル(1300-1600)は39.0s、中長距離(>1600)は40.2sで補正
    base_last3f = np.where(df['distance_num'] <= 1200, 37.5, np.where(df['distance_num'] <= 1600, 39.0, 40.2))

    speed_cols = []
    for i in range(1, 6):
        df[f'prev{i}_rank'] = pd.to_numeric(df.get(f'前走{i}_着順', '').astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        df[f'prev{i}_last3f'] = df.get(f'前走{i}_上がり3F', '').apply(parse_time_to_sec)
        df[f'prev{i}_1c'] = df.get(f'前走{i}_通過順', '').apply(parse_passing_first)
        
        # 着順に基づく基礎指数 (1着=95, 2着=88 ... 8着以下=30)
        rank_base = np.maximum(30.0, 95.0 - (df[f'prev{i}_rank'].fillna(10.0) - 1.0) * 7.0)
        # 距離に応じた基準上がり3F差分による補正
        last3f_bonus = np.where(df[f'prev{i}_last3f'].notna(), (base_last3f - df[f'prev{i}_last3f']) * 2.5, 0.0)
        last3f_bonus = np.clip(last3f_bonus, -10.0, 12.0)
        
        df[f'prev{i}_speed_idx'] = np.where(df[f'prev{i}_rank'].notna(), rank_base + last3f_bonus, np.nan)
        speed_cols.append(f'prev{i}_speed_idx')

    # 4. キャリア信頼度補正と欠損補完（キャリア1〜2戦の過大評価抑制）
    df['valid_runs_count'] = df[speed_cols].notna().sum(axis=1)
    df['is_no_past_data'] = (df['valid_runs_count'] == 0).astype(int)
    # キャリアが2戦以下の場合は信頼度係数を掛けて平均値(55pt)へシュリンク
    reliability = np.where(df['valid_runs_count'] == 0, 0.0,
                  np.where(df['valid_runs_count'] == 1, 0.65,
                  np.where(df['valid_runs_count'] == 2, 0.85, 1.0)))

    raw_mean_speed = df[speed_cols].mean(axis=1).fillna(55.0)
    df['horse_mean_speed'] = 55.0 + (raw_mean_speed - 55.0) * reliability
    df['horse_max_speed'] = 55.0 + (df[speed_cols].max(axis=1).fillna(55.0) - 55.0) * reliability
    df['horse_min_speed'] = df[speed_cols].min(axis=1).fillna(40.0)
    df['horse_std_speed'] = df[speed_cols].std(axis=1).fillna(5.0)

    last3f_cols = [f'prev{i}_last3f' for i in range(1, 6)]
    df['horse_mean_last3f'] = df[last3f_cols].mean(axis=1).fillna(40.0)
    df['horse_min_last3f'] = df[last3f_cols].min(axis=1).fillna(38.0)
    df['horse_max_last3f'] = df[last3f_cols].max(axis=1).fillna(42.0)
    df['horse_mean_1c'] = df[[f'prev{i}_1c' for i in range(1, 6)]].mean(axis=1).fillna(6.0)

    for i in range(1, 6):
        df[f'prev{i}_speed_idx_safe'] = df[f'prev{i}_speed_idx'].fillna(df['horse_mean_speed'])
        df[f'prev{i}_last3f_safe'] = df[f'prev{i}_last3f'].fillna(df['horse_mean_last3f'])
        df[f'prev{i}_1c_safe'] = df[f'prev{i}_1c'].fillna(df['horse_mean_1c'])

    # 5. 集約指数とトレンド
    df['ema3_custom_time_index_m'] = df['prev1_speed_idx_safe'] * 0.5 + df['prev2_speed_idx_safe'] * 0.3 + df['prev3_speed_idx_safe'] * 0.2
    df['ema5_custom_time_index_m'] = df['prev1_speed_idx_safe'] * 0.35 + df['prev2_speed_idx_safe'] * 0.25 + df['prev3_speed_idx_safe'] * 0.2 + df['prev4_speed_idx_safe'] * 0.1 + df['prev5_speed_idx_safe'] * 0.1
    df['best_custom_time_index_m'] = df[[f'prev{i}_speed_idx_safe' for i in range(1, 6)]].max(axis=1)
    df['ema3_custom_last3f_index'] = (60.0 - df['prev1_last3f_safe'] * 0.5 - df['prev2_last3f_safe'] * 0.3 - df['prev3_last3f_safe'] * 0.2) * 2.0
    df['ema3_custom_start_index'] = (14.0 - df['prev1_1c_safe'] * 0.5 - df['prev2_1c_safe'] * 0.3 - df['prev3_1c_safe'] * 0.2) * 3.0

    df['speed_trend_1_2'] = df['prev1_speed_idx_safe'] - df['prev2_speed_idx_safe']
    df['speed_trend_1_3'] = df['prev1_speed_idx_safe'] - df['prev3_speed_idx_safe']
    df['speed_trend_ema'] = df['prev1_speed_idx_safe'] - df['ema5_custom_time_index_m']
    df['last3f_trend_1_2'] = df['prev1_last3f_safe'] - df['prev2_last3f_safe']
    df['pos_trend_1_2'] = df['prev1_1c_safe'] - df['prev2_1c_safe']

    # 6. 騎手・調教師レート（リーク遮断）
    if is_train:
        def calc_expanding_rates(group_col, target_col, prior_succ, prior_tot):
            sub = df[['race_id_str', group_col, target_col]].copy()
            sub['prev_target'] = sub.groupby(group_col)[target_col].shift(1).fillna(0.0)
            sub['cum_succ'] = sub.groupby(group_col)['prev_target'].cumsum()
            sub['prev_count'] = sub.groupby(group_col)[target_col].cumcount()
            rate = (sub['cum_succ'] + prior_succ) / (sub['prev_count'] + prior_tot)
            return rate.sort_index()

        df['jockey_win_rate'] = calc_expanding_rates('騎手_clean', 'target_win', 1.0, 10.0).fillna(0.08)
        df['jockey_place_rate'] = calc_expanding_rates('騎手_clean', 'target_place', 2.0, 10.0).fillna(0.20)
        df['trainer_win_rate'] = calc_expanding_rates('調教師_clean', 'target_win', 1.0, 10.0).fillna(0.08)
        df['jockey_course_win_rate'] = calc_expanding_rates('j_place_key', 'target_win', 0.5, 5.0).fillna(df['jockey_win_rate'])
    else:
        if past_j_stats is not None: df = pd.merge(df, past_j_stats, on='騎手_clean', how='left')
        if past_t_stats is not None: df = pd.merge(df, past_t_stats, on='調教師_clean', how='left')
        if past_jp_stats is not None: df = pd.merge(df, past_jp_stats, on='j_place_key', how='left')

        df['jockey_win_rate'] = df.get('jockey_win_rate', pd.Series(dtype=float)).fillna(0.08)
        df['jockey_place_rate'] = df.get('jockey_place_rate', pd.Series(dtype=float)).fillna(0.20)
        df['trainer_win_rate'] = df.get('trainer_win_rate', pd.Series(dtype=float)).fillna(0.08)
        df['jockey_course_win_rate'] = df.get('jockey_course_win_rate', pd.Series(dtype=float)).fillna(df['jockey_win_rate'])

    # 7. 相互作用
    df['interaction_time_jockey_win'] = df['ema3_custom_time_index_m'] * df['jockey_win_rate']
    df['interaction_time_jockey_place'] = df['ema3_custom_time_index_m'] * df['jockey_place_rate']
    df['interaction_time_course_jockey'] = df['ema3_custom_time_index_m'] * df['jockey_course_win_rate']
    df['interaction_best_jockey'] = df['best_custom_time_index_m'] * df['jockey_place_rate']
    df['interaction_ema5_trainer'] = df['ema5_custom_time_index_m'] * df['trainer_win_rate']
    df['interaction_start_last3f'] = df['ema3_custom_start_index'] * df['ema3_custom_last3f_index']
    df['interaction_front_heavy'] = df['ema3_custom_start_index'] * (df['is_heavy_track'] + 1.0)
    df['interaction_style_waku'] = df['is_style_nige'] * (9.0 - df['waku_num']) + df['is_style_sashi'] * df['waku_num']
    df['interaction_weight_fatigue'] = df['kinryo_ratio'] * df['weight_change']
    df['interaction_weight_speed'] = df['ema3_custom_time_index_m'] / df['kinryo_ratio'].replace(0, np.nan).fillna(0.12)
    df['interaction_trainer_jockey'] = df['trainer_win_rate'] * df['jockey_win_rate']
    df['interaction_waku_jockey'] = df['waku_num'] * df['jockey_win_rate']

    # 8. レース内相対偏差
    rel_targets = [
        'body_weight', 'weight_change', 'kinryo_ratio', 'weight_change_ratio',
        'horse_mean_speed', 'horse_max_speed', 'horse_min_speed', 'horse_std_speed',
        'horse_mean_last3f', 'horse_min_last3f', 'horse_max_last3f', 'horse_mean_1c',
        'speed_trend_1_2', 'speed_trend_1_3', 'speed_trend_ema', 'last3f_trend_1_2', 'pos_trend_1_2',
        'ema3_custom_time_index_m', 'ema5_custom_time_index_m', 'best_custom_time_index_m',
        'ema3_custom_last3f_index', 'ema3_custom_start_index',
        'jockey_win_rate', 'jockey_place_rate', 'trainer_win_rate', 'jockey_course_win_rate',
        'interaction_time_jockey_win', 'interaction_time_jockey_place',
        'interaction_time_course_jockey', 'interaction_best_jockey',
        'interaction_ema5_trainer', 'interaction_start_last3f',
        'interaction_front_heavy', 'interaction_style_waku',
        'interaction_weight_fatigue', 'interaction_weight_speed',
        'interaction_trainer_jockey', 'interaction_waku_jockey'
    ]
    for i in range(1, 6):
        rel_targets.extend([f'prev{i}_speed_idx_safe', f'prev{i}_last3f_safe', f'prev{i}_1c_safe'])

    new_rel_cols = {}
    for col in rel_targets:
        if col not in df.columns: continue
        m = df.groupby('race_id_str')[col].transform('mean')
        s = df.groupby('race_id_str')[col].transform('std').replace(0, 1.0).fillna(1.0)
        ma = df.groupby('race_id_str')[col].transform('max').replace(0, 1.0)

        new_rel_cols[f'{col}_race_diff'] = df[col] - m
        new_rel_cols[f'{col}_race_zscore'] = (df[col] - m) / s
        asc = True if any(k in col for k in ['rank', 'last3f', '1c', 'mean_rank', 'trend']) else False
        new_rel_cols[f'{col}_race_rank'] = df.groupby('race_id_str')[col].rank(ascending=asc, method='min')
        new_rel_cols[f'{col}_race_mean_ratio'] = df[col] / m.replace(0, np.nan).fillna(1.0)
        new_rel_cols[f'{col}_race_max_ratio'] = df[col] / ma.replace(0, np.nan).fillna(1.0)

    df = pd.concat([df, pd.DataFrame(new_rel_cols, index=df.index)], axis=1)
    df = df.rename(columns=lambda x: re.sub(r'[^\w\s\-_]', '', x))
    return df