@echo off
chcp 65001 > nul
title 🌸 勝ち子ちゃん 全自動更新BOT

echo ==================================================
echo  🌸 勝ち子ちゃん 全自動更新スクリプトを起動します
echo ==================================================
echo.

cd /d "%~dp0"

echo ⏳ 過去データの同期・キャッシュ再構築・Git送信を実行中...
echo.

python update_all.py

if errorlevel 1 goto ERROR_END
goto SUCCESS_END

:ERROR_END
echo.
echo ==================================================
echo ❌ エラーが発生しました。処理を中断します。
echo 上記のエラーメッセージをご確認ください。
echo ==================================================
goto EXIT

:SUCCESS_END
echo.
echo ==================================================
echo ✨ 全ての更新処理＆Git Pushが正常に完了しました！
echo 🚀 Webアプリ(Streamlit)側で自動反映されます。
echo ==================================================
goto EXIT

:EXIT
echo.
pause