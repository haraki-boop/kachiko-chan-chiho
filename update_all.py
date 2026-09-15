import subprocess
import sys
import os
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"

print("==================================================")
print(f"🌸 勝ち子ちゃん 日次データ自動更新BOT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
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
                print(f"⚠️ [{description}] (変更がないためスキップしました)")
                return False
            else:
                print(f"❌ [{description}] がエラーで終了しました。(コード: {result.returncode})")
                sys.exit(1)
    except Exception as e:
        print(f"❌ [{description}] の実行中にエラーが発生しました。\n   詳細: {e}")
        if not ignore_error:
            sys.exit(1)
        return False

# ① 過去データ・結果の更新
if os.path.exists("update_results_chiho.py"):
    run_command(f"{PYTHON_EXE} update_results_chiho.py", "① 過去データの更新")

# ② 本日の出馬表取得
if os.path.exists("scrape_chiho_today.py"):
    run_command(f"{PYTHON_EXE} scrape_chiho_today.py", "② 本日出馬表の更新")

# 🚨 ここを追加！: キャッシュを作る前にAIを再学習させてスコアを出す
if os.path.exists("train_ensemble_model.py"):
    run_command(f"{PYTHON_EXE} train_ensemble_model.py", "②.5 AIモデルの再チューニング (NDCGスコア算出)")

# ③ アプリ用キャッシュ生成
if os.path.exists("create_cache_chiho.py"):
    run_command(f"{PYTHON_EXE} create_cache_chiho.py", "③ アプリ用予測キャッシュの生成")

# ④ Gitへのアップロード (必要なファイルだけ！)
run_command("git add future_races_chiho.csv app_cache_chiho.pkl", "④-1 アプリに必要なファイルだけをステージング")

commit_msg = f"auto: 日次データ更新 ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
run_command(f'git commit -m "{commit_msg}"', "④-2 Gitコミット", ignore_error=True)

run_command("git config --global http.postBuffer 1572864000", "Git通信設定の最適化", ignore_error=True)
run_command("git push origin main", "④-3 GitHubへ送信", ignore_error=True)

print("\n==================================================")
print("🎉 すべての更新処理が完了しました！")
print("==================================================")