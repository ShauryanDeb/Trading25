import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, APIError
import traceback

print('Script started!')

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

if not API_KEY or not API_SECRET:
    print('API keys not found. Please check your .env file.')
    exit(1)

# Initialize Alpaca REST API
api = REST(API_KEY, API_SECRET, BASE_URL)

try:
    account = api.get_account()
    if account:
        print('Account ID:', account.id)
        print('Status:', account.status)
        print('Equity:', account.equity)
        print('Buying Power:', account.buying_power)
        print('Cash:', account.cash)
        print('Portfolio Value:', account.portfolio_value)
    else:
        print('No account information returned.')
except APIError as e:
    print('Alpaca API error:', e)
except Exception as e:
    print('Unexpected error:')
    traceback.print_exc() 