import subprocess
import sys
import os
from datetime import datetime

print("==================================================")
print(f"🌸 勝ち子ちゃん 一括データ更新＆Git自動同期BOT ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("==================================================")

# 実行する処理のステップ定義
def run_command(cmd, description):
    print(f"\n🔄 [{description}] を実行中...")
    try:
        # コマンドの実行
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        print(f"✅ [{description}] 完了")
        if result.stdout.strip():
            print(f"   💬 出力:\n{result.stdout.strip()[:300]}") # 最初の300文字を表示
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [{description}] でエラーが発生しました。")
        print(f"   ⚠️ エラー内容:\n{e.stderr.strip()}")
        return False

# STEP 1: 未来レースデータ（出馬表）の取得スクリプト実行
# ※お持ちのスクリプト名（例: scrape_future.py 等）に合わせて変更してください
if os.path.exists("scrape_future.py"):
    if not run_command("python scrape_future.py", "最新出馬表データの取得・作成"):
        print("⚠️ 出馬表更新で問題が発生しましたが処理を継続します。")
else:
    print("ℹ️ scrape_future.py が見つからないためデータ取得ステップをスキップします。")

# STEP 2: Git ステージング (git add)
if not run_command("git add .", "変更ファイルのステージング (git add)"):
    sys.exit(1)

# STEP 3: Git コミット (git commit)
commit_msg = f"auto: レースデータおよび予測結果の一括自動更新 ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
run_command(f'git commit -m "{commit_msg}"', "Gitコミット (git commit)")

# STEP 4: Git プッシュ (git push)
if run_command("git push origin main", "リモートリポジトリへ送信 (git push)"):
    print("\n==================================================")
    print("🎉 すべての一括更新＆Git同期作業が正常に完了しました！")
    print("==================================================")
else:
    # mainで失敗した場合はmasterを試行
    print("🔄 origin/main での送信に失敗したため、origin/master を試行します...")
    if run_command("git push origin master", "リモートリポジトリへ送信 (git push origin master)"):
        print("\n==================================================")
        print("🎉 すべての一括更新＆Git同期作業が正常に完了しました！")
        print("==================================================")
    else:
        print("\n⚠️ Git Push に失敗しました。ネットワーク状態やコンフリクトを確認してください。")