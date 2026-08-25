import subprocess
import sys
import os
from datetime import datetime

print("==================================================")
print(f"🌸 勝ち子ちゃん 全自動更新＆Git同期BOT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("==================================================")

def run_command(cmd, description):
    print(f"\n🔄 [{description}] を実行中...")
    try:
        # エラーで落ちないよう check=False にし、文字コードエラーは強制置換(replace)で突破
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            errors='replace'
        )
        
        if result.returncode == 0:
            print(f"✅ [{description}] 完了")
            if result.stdout and result.stdout.strip():
                print(f"   💬 出力:\n{result.stdout.strip()[:300]}")
            return True
        else:
            print(f"⚠️ [{description}] で警告またはエラーが発生しました。")
            if result.stderr and result.stderr.strip():
                print(f"   詳細:\n{result.stderr.strip()[:300]}")
            elif result.stdout and result.stdout.strip():
                print(f"   詳細:\n{result.stdout.strip()[:300]}")
            return False
            
    except Exception as e:
        print(f"❌ [{description}] の実行中に予期せぬエラーが発生しました。")
        print(f"   詳細: {e}")
        return False

# STEP 1: 前日の確定結果＆的中履歴の更新
if os.path.exists("update_results_chiho.py"):
    run_command("python update_results_chiho.py", "確定着順＆結果データの更新")

# STEP 2: 本日の最新出馬表データのスクレイピング取得
if os.path.exists("scrape_chiho_today.py"):
    run_command("python scrape_chiho_today.py", "本日開催の最新出馬表取得")

# STEP 3: Git ステージング
if not run_command("git add .", "全変更ファイルのステージング (git add)"):
    print("❌ Git add に失敗しました。処理を中断します。")
    sys.exit(1)

# STEP 4: Git コミット
commit_msg = f"auto: 本日データ一括更新 ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
run_command(f'git commit -m "{commit_msg}"', "Gitコミット (git commit)")

# STEP 5: Git プッシュ
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
        print("\n❌ Git Push に失敗しました。コンフリクト等が発生している可能性があります。")