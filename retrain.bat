@echo off
cd /d "C:\Users\aweso\StockTrading\Trading25"
"C:\Users\aweso\anaconda3\python.exe" run.py train --universe --output models/model.pkl >> "C:\Users\aweso\StockTrading\Trading25\logs\retrain.log" 2>&1
"C:\Users\aweso\anaconda3\python.exe" trading_bots/intraday_bot.py --train >> "C:\Users\aweso\StockTrading\Trading25\logs\retrain.log" 2>&1
