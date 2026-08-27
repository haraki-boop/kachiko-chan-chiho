import subprocess
import sys
import os
from datetime import datetime

# 子プロセスの文字コードエラーを強制的に防ぐ
os.environ["PYTHONIOENCODING"] = "utf-8"

print("==================================================")
print(f"🌸 勝ち子ちゃん 全自動更新＆Git同期BOT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("==================================================")

# 実行中のPythonパスを確実に使用
PYTHON_EXE = f'"{sys.executable}"'

def run_command(cmd, description, stop_on_error=False):
    print(f"\n🔄 [{description}] を実行中...")
    try:
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode == 0:
            print(f"✅ [{description}] 完了")
            return True
        else:
            print(f"⚠️ [{description}] がエラーで終了しました。(コード: {result.returncode})")
            if stop_on_error:
                print(f"❌ 致命的なエラーのため、以降の同期処理を中断します。")
                sys.exit(1)
            return False
            
    except Exception as e:
        print(f"❌ [{description}] の実行中にエラーが発生しました。")
        print(f"   詳細: {e}")
        if stop_on_error:
            sys.exit(1)
        return False

# STEP 1: 前日の確定結果＆的中履歴の更新
if os.path.exists("update_results_chiho.py"):
    run_command(f"{PYTHON_EXE} update_results_chiho.py", "確定着順＆結果データの更新")

# STEP 2: 昨日の予想の答え合わせ・成績集計
if os.path.exists("evaluate_yesterday.py"):
    run_command(f"{PYTHON_EXE} evaluate_yesterday.py", "昨日の成績集計・答え合わせ")

# STEP 3: 新AI（第3形態）の再学習・脳みそアップデート（失敗時はPush中断）
if os.path.exists("train_ensemble_model.py"):
    run_command(f"{PYTHON_EXE} train_ensemble_model.py", "第3形態AIのアンサンブル学習(脳みそ更新)", stop_on_error=True)

# STEP 4: 本日の最新出馬表データのスクレイピング取得（失敗時はPush中断）
if os.path.exists("scrape_chiho_today.py"):
    run_command(f"{PYTHON_EXE} scrape_chiho_today.py", "本日開催の最新出馬表取得", stop_on_error=True)

# STEP 5: Git ステージング
if not run_command("git add .", "全変更ファイルのステージング (git add)"):
    print("❌ Git add に失敗しました。処理を中断します。")
    sys.exit(1)

# STEP 6: Git コミット
commit_msg = f"auto: 本日データ一括更新 ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
run_command(f'git commit -m "{commit_msg}"', "Gitコミット (git commit)")

# STEP 7: Git プッシュ
if run_command("git push origin main", "リモート(main)へ送信 (git push)"):
    print("\n==================================================")
    print("🎉 すべての更新＆Webアプリへの自動同期が完了しました！")
    print("==================================================")
else:
    print("🔄 origin/master での送信を再試行中...")
    if run_command("git push origin master", "リモート(master)へ送信"):
        print("\n==================================================")
        print("🎉 すべての更新＆Webアプリへの自動同期が完了しました！")
        print("==================================================")
    else:
        print("\n❌ Git Push に失敗しました。")