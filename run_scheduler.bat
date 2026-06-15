@echo off
cd /d "C:\Users\aweso\StockTrading\Trading25"
"C:\Users\aweso\anaconda3\python.exe" trading_bots/scheduler.py --run-now >> "C:\Users\aweso\StockTrading\Trading25\logs\scheduler.log" 2>&1
