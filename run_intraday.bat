@echo off
cd /d "C:\Users\aweso\StockTrading\Trading25"
set CONDA_PREFIX=C:\Users\aweso\anaconda3
set PYTHONPATH=C:\Users\aweso\anaconda3\Lib\site-packages;C:\Users\aweso\anaconda3\Lib;C:\Users\aweso\anaconda3
set PATH=C:\Users\aweso\anaconda3;C:\Users\aweso\anaconda3\Scripts;C:\Users\aweso\anaconda3\Library\bin;%PATH%
"C:\Users\aweso\anaconda3\python.exe" trading_bots/intraday_bot.py >> logs\intraday.log 2>&1
