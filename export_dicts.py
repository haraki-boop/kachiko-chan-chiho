import os
import re
import pandas as pd
import numpy as np
import joblib
import unicodedata

print("📦 巨大な履歴データから、アプリ用の軽量辞書を抽出しています...")

# V2データを優先読み込み
target_csv = "ml_target_data_chiho.csv"
for f in ["ml_target_data_chiho_v2.csv", "ml_target_data_v2.csv", "ml_target_data_chiho.csv"]:
    if os.path.exists(f) and os.path.getsize(f) > 100:
        target_csv = f
        break

try:
    df_p = pd.read_csv(target_csv, low_memory=False, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_p = pd.read_csv(target_csv, low_memory=False, encoding='cp932')

def clean_horse_name(name_val): 
    if pd.isna(name_val): return ""
    s_val = unicodedata.normalize('NFKC', str(name_val))
    return re.sub(r'[\s・･._\u3000\t\r\n]+', '', s_val).strip()

def parse_rank(x):
    if pd.isna(x): return np.nan
    s_val = str(x).replace('着', '').replace('(', '').replace(')', '').strip()
    try: return float(s_val)
    except: return np.nan

def parse_time_str(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r'(?:(\d+)[:.])?(\d{1,2})\.(\d+)', s)
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        return mins * 60 + int(m.group(2)) + float('0.' + m.group(3))
    try: return float(s)
    except: return np.nan

df_p['馬名_clean'] = df_p['馬名'].astype(str).apply(clean_horse_name)
rank_col = '着順_num' if '着順_num' in df_p.columns else '着順'
df_p['target_rank_tmp'] = df_p[rank_col].apply(parse_rank)
df_p = df_p[df_p['target_rank_tmp'].notna() & (df_p['target_rank_tmp'] < 90.0)].copy()

df_p['target_win'] = (df_p['target_rank_tmp'] == 1.0).astype(int)
df_p['target_rentai'] = (df_p['target_rank_tmp'] <= 2.0).astype(int)
df_p['time_sec_clean'] = df_p.get('タイム', df_p.get('time')).apply(parse_time_str)
df_p['last3f_sec_clean'] = df_p.get('上り', df_p.get('上がり3F')).apply(parse_time_str)
df_p['last_3f'] = df_p['last3f_sec_clean'].fillna(39.0)
df_p['first_corner_raw'] = pd.to_numeric(df_p.get('first_corner', df_p.get('1角')), errors='coerce').fillna(5.0)
df_p['last_corner_raw'] = pd.to_numeric(df_p.get('last_corner', df_p.get('4角')), errors='coerce').fillna(df_p['first_corner_raw'])
df_p['custom_time_index'] = pd.to_numeric(df_p.get('custom_time_index'), errors='coerce')
df_p['custom_start_index'] = pd.to_numeric(df_p.get('custom_start_index'), errors='coerce')

df_p['騎手_clean'] = df_p.get('騎手', pd.Series(['']*len(df_p))).astype(str).apply(clean_horse_name)
df_p['trainer_clean'] = df_p.get('調教師', df_p['騎手_clean']).astype(str).apply(clean_horse_name)
df_p['jockey_trainer_combo'] = df_p['騎手_clean'] + "_" + df_p['trainer_clean']
df_p['place_code_tmp'] = df_p['race_id'].astype(str).str[4:6]
df_p['prize_num'] = pd.to_numeric(df_p.get('賞金(万円)', 0), errors='coerce').fillna(0.0)
df_p['prize_num_log'] = np.log1p(df_p['prize_num'])

jockey_dict = df_p.groupby('騎手_clean')['target_win'].mean().to_dict()
jockey_rentai_dict = df_p.groupby('騎手_clean')['target_rentai'].mean().to_dict()
trainer_dict = df_p.groupby('trainer_clean')['target_win'].mean().to_dict()
combo_dict = df_p.groupby('jockey_trainer_combo')['target_win'].mean().to_dict()
place_avg_rank_dict = df_p.groupby(['馬名_clean', 'place_code_tmp'])['target_rank_tmp'].mean().to_dict()

df_p['waku_num_tmp'] = pd.to_numeric(df_p.get('枠番'), errors='coerce').fillna(0)
df_p['place_waku_combo'] = df_p['place_code_tmp'] + "_" + df_p['waku_num_tmp'].astype(str)
waku_dict = df_p.groupby('place_waku_combo')['target_win'].mean().to_dict()

df_p['date_dt'] = pd.to_datetime(df_p.get('date'), errors='coerce').fillna(pd.to_datetime('2020-01-01'))
df_p = df_p.sort_values(['馬名_clean', 'date_dt']).reset_index(drop=True)

sorted_p = df_p.sort_values('date_dt')
def calc_ema(series, span):
    s = series.dropna()
    if s.empty: return np.nan
    return s.ewm(span=span, min_periods=1).mean().iloc[-1]

horse_dict = {}
for h, group in sorted_p.groupby('馬名_clean'):
    r3 = group.tail(3)
    r5 = group.tail(5)
    last_row = group.iloc[-1]
    
    f_c = calc_ema(r3['first_corner_raw'], 3)
    l_c = calc_ema(r3['last_corner_raw'], 3)
    time_idx_avg = calc_ema(r3['custom_time_index'], 3)
    start_idx_avg = calc_ema(r3['custom_start_index'], 3)
    
    horse_dict[h] = {
        'first_corner': f_c if pd.notna(f_c) else 5.0,
        'last_corner': l_c if pd.notna(l_c) else 5.0,
        'horse_prize_avg': calc_ema(r5['prize_num_log'], 5) if pd.notna(calc_ema(r5['prize_num_log'], 5)) else 0.0,
        'prev_time_index_avg': time_idx_avg if pd.notna(time_idx_avg) else 100.0,
        'prev_start_index_avg': start_idx_avg if pd.notna(start_idx_avg) else 50.0,
        'prev_time_sec': last_row.get('time_sec_clean', 90.0),
        'prev_last3f_sec': last_row.get('last3f_sec_clean', 39.0),
        'horse_career_runs': len(group),
        'prev_jockey_win': jockey_dict.get(last_row.get('騎手_clean', ''), 0.05),
        'prev_prize_log': last_row.get('prize_num_log', 0.0),
        'is_minami_kanto': last_row.get('is_minami_kanto', 0)
    }

export_data = {
    'jockey_dict': jockey_dict,
    'horse_dict': horse_dict,
    'waku_dict': waku_dict,
    'trainer_dict': trainer_dict,
    'combo_dict': combo_dict,
    'place_avg_rank_dict': place_avg_rank_dict,
    'jockey_rentai_dict': jockey_rentai_dict
}

joblib.dump(export_data, 'past_dicts.pkl')
print(f"✨ 完了！軽量辞書データ 'past_dicts.pkl' (約{os.path.getsize('past_dicts.pkl')/1024/1024:.2f} MB) を作成しました！")