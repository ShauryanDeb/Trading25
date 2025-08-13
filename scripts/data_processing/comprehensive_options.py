import argparse
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Enhanced option features for comprehensive analysis
OPTION_FEATURES = [
    # Volume and Open Interest
    "CallVolume", "PutVolume", "CallOI", "PutOI",
    "TotalVolume", "TotalOI", "PCR_Volume", "PCR_OI",
    
    # Implied Volatility
    "CallIV", "PutIV", "AvgIV", "IV_Skew",
    
    # Greeks (if available)
    "CallDelta", "PutDelta", "CallGamma", "PutGamma",
    "CallTheta", "PutTheta", "CallVega", "PutVega",
    
    # Strike Analysis
    "ATM_CallIV", "ATM_PutIV", "ITM_CallIV", "OTM_PutIV",
    "Strike_Range", "Strike_Count",
    
    # Market Sentiment
    "CallPut_Ratio", "IV_Premium", "Skew_Index",
    
    # Expiration Analysis
    "DaysToExpiry", "Expiry_Type", "NextExpiry_Days"
]

def get_all_expirations(ticker: str) -> list:
    """Get all available expiration dates for a ticker."""
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            raise ValueError(f"No option expirations found for {ticker}")
        return sorted(expirations)
    except Exception as e:
        print(f"Error getting expirations for {ticker}: {e}")
        return []

def calculate_atm_strike(spot_price: float, strikes: list) -> float:
    """Find the closest strike to current spot price (ATM)."""
    if not strikes:
        return spot_price
    return min(strikes, key=lambda x: abs(x - spot_price))

def calculate_greeks_approximation(spot: float, strike: float, iv: float, 
                                 time_to_expiry: float, option_type: str) -> dict:
    """Approximate Greeks using Black-Scholes-like calculations."""
    try:
        # Simplified Greeks calculation
        moneyness = np.log(spot / strike)
        time_sqrt = np.sqrt(time_to_expiry / 365)
        
        if option_type == 'call':
            delta = 0.5 + 0.5 * np.tanh(moneyness / (iv * time_sqrt))
            gamma = np.exp(-moneyness**2 / (2 * iv**2 * time_to_expiry)) / (spot * iv * time_sqrt * np.sqrt(2 * np.pi))
        else:  # put
            delta = -0.5 + 0.5 * np.tanh(-moneyness / (iv * time_sqrt))
            gamma = np.exp(-moneyness**2 / (2 * iv**2 * time_to_expiry)) / (spot * iv * time_sqrt * np.sqrt(2 * np.pi))
        
        theta = -iv * spot * gamma / 2
        vega = spot * time_sqrt * gamma
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def fetch_comprehensive_options_data(ticker: str, date: str = None, 
                                   expiration: str = None) -> pd.DataFrame:
    """Fetch comprehensive options data for analysis."""
    try:
        t = yf.Ticker(ticker)
        
        # Get current stock price
        hist = t.history(period="1d")
        if hist.empty:
            return pd.DataFrame()
        spot_price = hist['Close'].iloc[-1]
        
        # Get all expirations if not specified
        if expiration is None:
            expirations = get_all_expirations(ticker)
            if not expirations:
                return pd.DataFrame()
            expiration = expirations[0]  # Use nearest expiration
        
        # Fetch option chain
        chain = t.option_chain(expiration)
        calls = chain.calls
        puts = chain.puts
        
        if calls.empty and puts.empty:
            return pd.DataFrame()
        
        # Calculate days to expiry
        expiry_date = pd.to_datetime(expiration)
        current_date = pd.to_datetime(date) if date else pd.to_datetime('today')
        days_to_expiry = (expiry_date - current_date).days
        
        # Basic aggregations
        call_volume = calls['volume'].sum() if not calls.empty else 0
        put_volume = puts['volume'].sum() if not puts.empty else 0
        call_oi = calls['openInterest'].sum() if not calls.empty else 0
        put_oi = puts['openInterest'].sum() if not puts.empty else 0
        
        # Implied Volatility analysis
        call_iv = calls['impliedVolatility'].mean() if not calls.empty else 0
        put_iv = puts['impliedVolatility'].mean() if not puts.empty else 0
        
        # ATM strike analysis
        all_strikes = list(calls['strike']) + list(puts['strike'])
        atm_strike = calculate_atm_strike(spot_price, all_strikes)
        
        # Find ATM options
        atm_calls = calls[abs(calls['strike'] - atm_strike) < 1] if not calls.empty else pd.DataFrame()
        atm_puts = puts[abs(puts['strike'] - atm_strike) < 1] if not puts.empty else pd.DataFrame()
        
        atm_call_iv = atm_calls['impliedVolatility'].mean() if not atm_calls.empty else call_iv
        atm_put_iv = atm_puts['impliedVolatility'].mean() if not atm_puts.empty else put_iv
        
        # ITM/OTM analysis
        itm_calls = calls[calls['strike'] < spot_price] if not calls.empty else pd.DataFrame()
        otm_puts = puts[puts['strike'] > spot_price] if not puts.empty else pd.DataFrame()
        
        itm_call_iv = itm_calls['impliedVolatility'].mean() if not itm_calls.empty else call_iv
        otm_put_iv = otm_puts['impliedVolatility'].mean() if not otm_puts.empty else put_iv
        
        # Greeks calculation (simplified)
        if not calls.empty:
            sample_call = calls.iloc[0]
            call_greeks = calculate_greeks_approximation(
                spot_price, sample_call['strike'], sample_call['impliedVolatility'],
                days_to_expiry, 'call'
            )
        else:
            call_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
        if not puts.empty:
            sample_put = puts.iloc[0]
            put_greeks = calculate_greeks_approximation(
                spot_price, sample_put['strike'], sample_put['impliedVolatility'],
                days_to_expiry, 'put'
            )
        else:
            put_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        # Compile comprehensive data
        data = {
            # Basic metrics
            "CallVolume": call_volume,
            "PutVolume": put_volume,
            "CallOI": call_oi,
            "PutOI": put_oi,
            "TotalVolume": call_volume + put_volume,
            "TotalOI": call_oi + put_oi,
            "PCR_Volume": put_volume / call_volume if call_volume > 0 else 0,
            "PCR_OI": put_oi / call_oi if call_oi > 0 else 0,
            
            # IV metrics
            "CallIV": call_iv,
            "PutIV": put_iv,
            "AvgIV": (call_iv + put_iv) / 2 if call_iv > 0 and put_iv > 0 else max(call_iv, put_iv),
            "IV_Skew": put_iv - call_iv,
            "ATM_CallIV": atm_call_iv,
            "ATM_PutIV": atm_put_iv,
            "ITM_CallIV": itm_call_iv,
            "OTM_PutIV": otm_put_iv,
            
            # Greeks
            "CallDelta": call_greeks['delta'],
            "PutDelta": put_greeks['delta'],
            "CallGamma": call_greeks['gamma'],
            "PutGamma": put_greeks['gamma'],
            "CallTheta": call_greeks['theta'],
            "PutTheta": put_greeks['theta'],
            "CallVega": call_greeks['vega'],
            "PutVega": put_greeks['vega'],
            
            # Strike analysis
            "Strike_Range": max(all_strikes) - min(all_strikes) if all_strikes else 0,
            "Strike_Count": len(all_strikes),
            
            # Market sentiment
            "CallPut_Ratio": call_volume / put_volume if put_volume > 0 else 0,
            "IV_Premium": (call_iv + put_iv) / 2 - 0.2,  # Assuming 20% as baseline
            "Skew_Index": (put_iv - call_iv) / ((call_iv + put_iv) / 2) if (call_iv + put_iv) > 0 else 0,
            
            # Expiration info
            "DaysToExpiry": days_to_expiry,
            "Expiry_Type": "Weekly" if days_to_expiry <= 7 else "Monthly" if days_to_expiry <= 30 else "Quarterly",
            "Spot_Price": spot_price,
            "ATM_Strike": atm_strike
        }
        
        df = pd.DataFrame([data])
        df.index = [current_date]
        return df
        
    except Exception as e:
        print(f"Error fetching options data for {ticker}: {e}")
        return pd.DataFrame()

def collect_historical_options_data(ticker: str, start_date: str = None, 
                                  end_date: str = None, max_days: int = 30) -> pd.DataFrame:
    """Collect historical options data for multiple dates."""
    print(f"Collecting historical options data for {ticker}...")
    
    # Determine date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=max_days)).strftime('%Y-%m-%d')
    
    # Get all expirations
    expirations = get_all_expirations(ticker)
    if not expirations:
        print(f"No expirations found for {ticker}")
        return pd.DataFrame()
    
    print(f"Found {len(expirations)} expiration dates: {expirations[:5]}...")
    
    all_data = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"Processing {date_str}...")
        
        # Try each expiration date
        for expiration in expirations:
            try:
                df = fetch_comprehensive_options_data(ticker, date_str, expiration)
                if not df.empty:
                    df['Expiration'] = expiration
                    all_data.append(df)
                    break  # Use first successful expiration
            except Exception as e:
                continue
        
        current_date += timedelta(days=1)
        time.sleep(0.1)  # Rate limiting
    
    if all_data:
        result = pd.concat(all_data, axis=0)
        result = result.sort_index()
        print(f"Collected {len(result)} data points")
        return result
    else:
        print("No data collected")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive options data collection")
    parser.add_argument("ticker", help="Underlying ticker symbol")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--max-days", type=int, default=30, help="Maximum days to collect (if no start date)")
    parser.add_argument("--output", default="comprehensive_options.csv", help="Output CSV file")
    parser.add_argument("--single-date", help="Collect data for single date YYYY-MM-DD")
    parser.add_argument("--expiration", help="Specific expiration date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    if args.single_date:
        # Single date collection
        df = fetch_comprehensive_options_data(args.ticker, args.single_date, args.expiration)
    else:
        # Historical collection
        df = collect_historical_options_data(args.ticker, args.start, args.end, args.max_days)
    
    if not df.empty:
        df.to_csv(args.output)
        print(f"Saved comprehensive options data to {args.output}")
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Features: {list(df.columns)}")
    else:
        print("No data collected")

if __name__ == "__main__":
    main() 