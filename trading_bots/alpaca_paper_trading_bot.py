import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, APIError
import traceback

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
    print('Placing a sample market order: Buy 1 share of AAPL...')
    order = api.submit_order(
        symbol='AAPL',
        qty=1,
        side='buy',
        type='market',
        time_in_force='gtc'
    )
    print('Order submitted! ID:', order.id)
except APIError as e:
    print('Alpaca API error while placing order:', e)
except Exception as e:
    print('Unexpected error while placing order:')
    traceback.print_exc()

print('\nOpen positions:')
try:
    positions = api.list_positions()
    if positions:
        for pos in positions:
            print(f"{pos.symbol}: {pos.qty} shares @ avg price {pos.avg_entry_price}")
    else:
        print('No open positions.')
except Exception as e:
    print('Error fetching positions:')
    traceback.print_exc()

print('\nRecent orders:')
try:
    orders = api.list_orders(status='all', limit=5)
    for o in orders:
        print(f"{o.symbol}: {o.side} {o.qty} {o.type} status={o.status} submitted_at={o.submitted_at}")
except Exception as e:
    print('Error fetching orders:')
    traceback.print_exc() 