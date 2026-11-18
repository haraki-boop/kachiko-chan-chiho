import subprocess
import sys
import os
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

print("==================================================")
print(f"🌸 勝ち子ちゃん 日次自動パイプライン ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("==================================================")

PYTHON_EXE = f'"{sys.executable}"'

def run_command(cmd, description, ignore_error=False):
    print(f"\n🔄 [{description}] を実行中...")
    try:
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            print(f"✅ [{description}] 完了")
            return True
        else:
            if ignore_error:
                print(f"⚠️ [{description}] スキップまたは無視")
                return False
            else:
                print(f"❌ [{description}] エラー停止 (コード: {result.returncode})")
                sys.exit(1)
    except Exception as e:
        print(f"❌ [{description}] 実行例外: {e}")
        if not ignore_error:
            sys.exit(1)
        return False

# ① 過去データの差分取得（最新結果をデータベースへ追記）
if os.path.exists("update_v2_database.py"):
    run_command(f"{PYTHON_EXE} update_v2_database.py", "① 過去データベース差分更新")
else:
    print("⚠️ 'update_v2_database.py' が見つかりません。")

# ② 特徴量錬成 (リーク完全遮断データの作成・ml_target_data_chiho.csv更新)
if os.path.exists("create_rich_features.py"):
    run_command(f"{PYTHON_EXE} create_rich_features.py", "② 特徴量データの完全生成")
else:
    print("⚠️ 'create_rich_features.py' が見つかりません。")

# ③ AIモデルの再チューニング＆再学習（Optuna探索＋3大Ranker更新）
if os.path.exists("train_ensemble_model.py"):
    run_command(f"{PYTHON_EXE} train_ensemble_model.py", "③ アンサンブルAIモデルの再学習")
else:
    print("⚠️ 'train_ensemble_model.py' が見つかりません。")

# ④ 本日出馬表の取得（最新レース情報・近5走ログの取得）
if os.path.exists("scrape_chiho_today.py"):
    run_command(f"{PYTHON_EXE} scrape_chiho_today.py", "④ 本日出馬表の更新")
else:
    print("❌ 'scrape_chiho_today.py' が見つかりません。")
    sys.exit(1)

# ⑤ 予測推論＆アプリ用買い目キャッシュ生成（最新モデル×当日出馬表）
if os.path.exists("create_cache_chiho.py"):
    run_command(f"{PYTHON_EXE} create_cache_chiho.py", "⑤ アプリ用予測キャッシュの生成")
else:
    print("❌ 'create_cache_chiho.py' が見つかりません。")
    sys.exit(1)

# ⑥ Git自動コミット・プッシュ（当日推論成果物をGitHubへ強制同期）
run_command("git add future_races_chiho.csv app_cache_chiho.pkl", "⑥-1 必要ファイルのステージング")

commit_msg = f"auto: 日次自動更新 ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
run_command(f'git commit -m "{commit_msg}"', "⑥-2 Gitコミット", ignore_error=True)

run_command("git config --global http.postBuffer 1572864000", "Git通信設定の最適化", ignore_error=True)

# 💡 リモートの履歴不整合を無視してローカルの最新成果物で強制上書き（non-fast-forward 対策）
run_command("git push -f origin main", "⑥-3 GitHubへ強制送信")

print("\n==================================================")
print("🎉 パイプライン全体の自動更新処理が正常完了しました！")
print("==================================================")