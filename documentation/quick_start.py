#!/usr/bin/env python3
"""
Quick Start Guide for Trading Model Scripts

This script demonstrates the basic workflow using the organized script structure.
"""

import sys
import os
import pandas as pd
import joblib

# Add the script directories to the path
sys.path.extend([
    os.path.join(os.path.dirname(__file__), 'data_processing'),
    os.path.join(os.path.dirname(__file__), 'model_training'),
    os.path.join(os.path.dirname(__file__), 'backtesting'),
    os.path.join(os.path.dirname(__file__), 'evaluation'),
    os.path.join(os.path.dirname(__file__), 'trading')
])

def quick_start_demo():
    """Demonstrate the basic workflow using the organized scripts."""
    
    print("🚀 Trading Model Quick Start Demo")
    print("=" * 50)
    
    # Check if we have the required data files
    data_files = [
        '../StockCSVs/apple.csv',
        '../StockCSVs/apple_validation.csv',
        '../enhanced_apple_model.pkl'
    ]
    
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure you have the required data files in the correct locations.")
        return
    
    print("✅ All required files found!")
    
    try:
        # Step 1: Data Processing
        print("\n📊 Step 1: Data Processing")
        print("-" * 30)
        
        # Import data processing modules
        from data_processing import enhanced_features
        from data_processing import data_validation
        
        # Load and engineer features
        print("Loading and engineering features...")
        X, y, feature_names = enhanced_features.load_and_engineer_features(
            price_path='../StockCSVs/apple.csv',
            options_path='../StockCSVs/apple_validation.csv',
            use_comprehensive_options=True
        )
        
        print(f"✅ Features engineered: {len(feature_names)} features, {len(X)} samples")
        
        # Step 2: Model Training (if needed)
        print("\n🤖 Step 2: Model Training")
        print("-" * 30)
        
        # Check if model exists
        if os.path.exists('../enhanced_apple_model.pkl'):
            print("Loading existing model...")
            model_data = joblib.load('../enhanced_apple_model.pkl')
            model = model_data['model']
            print("✅ Model loaded successfully")
        else:
            print("Training new model...")
            from model_training import enhanced_train_model
            
            # Split data for training
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model, results = enhanced_train_model.train_model(
                X_train, y_train, feature_names,
                model_type="random_forest",
                n_estimators=100
            )
            print("✅ Model trained successfully")
        
        # Step 3: Model Evaluation
        print("\n📈 Step 3: Model Evaluation")
        print("-" * 30)
        
        from evaluation import enhanced_evaluate_model
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Evaluating model performance...")
        eval_results = enhanced_evaluate_model.comprehensive_evaluation(
            model, X_test, y_test,
            feature_names=feature_names,
            include_plots=False  # Set to True if you want plots
        )
        
        print(f"✅ Model accuracy: {eval_results['metrics']['accuracy']:.3f}")
        
        # Step 4: Backtesting
        print("\n📊 Step 4: Backtesting")
        print("-" * 30)
        
        from backtesting import enhanced_backtest
        
        print("Running backtest...")
        enhanced_backtest.run_backtest(
            model_path='../enhanced_apple_model.pkl',
            price_file='../StockCSVs/apple.csv',
            options_file='../StockCSVs/apple_validation.csv',
            threshold=0.55
        )
        
        print("✅ Backtest completed")
        
        # Step 5: Trading Simulation
        print("\n🚀 Step 5: Trading Simulation")
        print("-" * 30)
        
        print("Note: Real-time trading requires Alpaca API credentials")
        print("To run live trading, use: python trading/realtime.py")
        
        print("\n🎉 Quick start demo completed successfully!")
        print("\nNext steps:")
        print("1. Review the README files in each directory")
        print("2. Experiment with different model parameters")
        print("3. Test with different stocks and time periods")
        print("4. Set up Alpaca API for live trading")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install pandas numpy scikit-learn joblib alpaca-trade-api backtrader")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Check that all files are in the correct locations and formats.")

def show_directory_structure():
    """Show the organized directory structure."""
    
    print("\n📁 Organized Directory Structure")
    print("=" * 50)
    
    structure = """
scripts/
├── 📊 data_processing/          # Data loading, validation, feature engineering
│   ├── data.py                  # Basic data loading
│   ├── features.py              # Core technical indicators
│   ├── enhanced_features.py     # Advanced features with options
│   ├── data_validation.py       # Data quality checks
│   └── ... (11 files total)
│
├── 🤖 model_training/           # Model training and optimization
│   ├── train_model.py           # Basic model training
│   ├── enhanced_train_model.py  # Advanced training with options
│   ├── tune_model.py            # Hyperparameter tuning
│   ├── ensemble_learning.py     # Ensemble methods
│   └── ... (13 files total)
│
├── 📈 backtesting/              # Strategy backtesting and analysis
│   ├── backtest.py              # Basic backtesting
│   ├── enhanced_backtest.py     # Advanced backtesting
│   ├── advanced_backtest.py     # Realistic market simulation
│   ├── advanced_risk_manager.py # Risk management
│   └── ... (8 files total)
│
├── 📊 evaluation/               # Model evaluation and metrics
│   ├── evaluate_model.py        # Basic evaluation
│   ├── enhanced_evaluate_model.py # Comprehensive evaluation
│   ├── walk_forward_testing.py  # Walk-forward analysis
│   └── ... (5 files total)
│
├── 🚀 trading/                  # Live trading and execution
│   ├── realtime.py              # Real-time trading bot
│   └── multi_stock_test.py      # Multi-stock trading
│
├── README.md                    # Main documentation
└── quick_start.py               # This file
"""
    
    print(structure)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--structure":
        show_directory_structure()
    else:
        quick_start_demo()
        show_directory_structure() 