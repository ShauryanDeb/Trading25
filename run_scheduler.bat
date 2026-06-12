@echo off
cd /d "C:\Users\aweso\Documents\TradingModel\Trading25"
"C:\Users\aweso\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe" trading_bots/scheduler.py --test-now >> logs\scheduler.log 2>&1
