# Data Processing Scripts

This directory contains scripts for data loading, validation, feature engineering, and preprocessing.

## Core Files

### Data Loading
- **`data.py`** - Basic data loading utilities and functions
- **`data_validation.py`** - Data quality checks and validation
- **`enhanced_data_validation.py`** - Comprehensive data validation with advanced checks

### Feature Engineering
- **`features.py`** - Core technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **`enhanced_features.py`** - Advanced feature engineering with options support
- **`advanced_feature_engineering.py`** - Sophisticated feature creation and engineering
- **`market_microstructure_features.py`** - Market microstructure indicators
- **`vectorize.py`** - Feature vectorization utilities

### Options Data
- **`options_data.py`** - Basic options data processing utilities
- **`comprehensive_options.py`** - Advanced options feature engineering

### Utilities
- **`add_target_to_features.py`** - Target variable creation and labeling

## Usage Examples

### Basic Data Loading
```python
from data_processing.data import load_stock_data
df = load_stock_data('AAPL.csv')
```

### Feature Engineering
```python
from data_processing.features import add_technical_indicators
df = add_technical_indicators(df)
```

### Enhanced Features with Options
```python
from data_processing.enhanced_features import load_and_engineer_features
X, y, feature_names = load_and_engineer_features(
    price_path='AAPL.csv',
    options_path='AAPL_options.csv',
    use_comprehensive_options=True
)
```

### Data Validation
```python
from data_processing.data_validation import validate_data
validation_results = validate_data(df)
```

## Feature Types

### Technical Indicators
- Moving averages (SMA, EMA)
- Bollinger Bands
- MACD
- RSI
- Stochastic Oscillator
- ATR (Average True Range)
- CCI (Commodity Channel Index)
- OBV (On-Balance Volume)

### Options Features
- Call/Put volume ratios
- Implied volatility metrics
- Options flow indicators
- Greeks-based features

### Market Microstructure
- Order book features
- Volume profile indicators
- Price impact measures
- Liquidity metrics

## Data Validation Checks

- Missing data detection
- Outlier identification
- Data type validation
- Date range verification
- Price consistency checks
- Volume validation
- Options data alignment 