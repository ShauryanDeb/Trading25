import joblib
import pandas as pd
import numpy as np
import os

def test_expanded_model(model_file="expanded_multi_stock_model.pkl"):
    """Test the expanded multi-stock model."""
    
    if not os.path.exists(model_file):
        print(f"❌ Model file {model_file} not found!")
        return
    
    print(f"🔍 Loading expanded multi-stock model: {model_file}")
    
    try:
        # Load the model
        model_data = joblib.load(model_file)
        
        print("\n" + "="*60)
        print("EXPANDED MULTI-STOCK MODEL ANALYSIS")
        print("="*60)
        
        # Model info
        print(f"\n📊 Model Information:")
        print(f"  Training Date: {model_data.get('training_date', 'Unknown')}")
        print(f"  Successful Stocks: {len(model_data.get('successful_stocks', []))}")
        print(f"  Selected Features: {len(model_data.get('selected_features', []))}")
        
        # Cross-validation results
        cv_scores = model_data.get('cv_scores', {})
        if cv_scores:
            print(f"\n🎯 Cross-Validation Results:")
            for name, scores in cv_scores.items():
                if isinstance(scores, dict):
                    print(f"  {name}: {scores['mean']:.4f} (+/- {scores['std'] * 2:.4f})")
                else:
                    print(f"  {name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        # Stock performance
        stock_performance = model_data.get('stock_performance', {})
        if stock_performance:
            accuracies = [perf['accuracy'] for perf in stock_performance.values()]
            print(f"\n📈 Stock Performance Summary:")
            print(f"  Average Accuracy: {np.mean(accuracies):.4f}")
            print(f"  Accuracy Std Dev: {np.std(accuracies):.4f}")
            print(f"  Best Stock: {max(stock_performance.items(), key=lambda x: x[1]['accuracy'])[0]}")
            print(f"  Worst Stock: {min(stock_performance.items(), key=lambda x: x[1]['accuracy'])[0]}")
            print(f"  Total Stocks: {len(stock_performance)}")
            
            # Top performing stocks
            top_stocks = sorted(stock_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:10]
            print(f"\n🏆 Top 10 Performing Stocks:")
            for i, (stock, perf) in enumerate(top_stocks, 1):
                print(f"  {i:2d}. {stock}: {perf['accuracy']:.4f} ({perf['samples']} samples)")
            
            # Bottom performing stocks
            bottom_stocks = sorted(stock_performance.items(), key=lambda x: x[1]['accuracy'])[:10]
            print(f"\n📉 Bottom 10 Performing Stocks:")
            for i, (stock, perf) in enumerate(bottom_stocks, 1):
                print(f"  {i:2d}. {stock}: {perf['accuracy']:.4f} ({perf['samples']} samples)")
        
        # Feature importance
        feature_importance = model_data.get('feature_importance', {})
        if feature_importance:
            print(f"\n🔍 Feature Importance Summary:")
            for name, importance_df in feature_importance.items():
                if not importance_df.empty:
                    top_features = importance_df.head(5)['feature'].tolist()
                    print(f"  {name} top features: {top_features}")
        
        # Model configuration
        model_config = model_data.get('model_config', {})
        if model_config:
            print(f"\n⚙️ Model Configuration:")
            for key, value in model_config.items():
                print(f"  {key}: {value}")
        
        print(f"\n✅ Expanded multi-stock model analysis complete!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

def compare_models():
    """Compare the original vs expanded models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    models = {
        'Original (5 stocks)': 'multi_stock_ensemble_model.pkl',
        'Expanded (79 stocks)': 'expanded_multi_stock_model.pkl'
    }
    
    for name, file in models.items():
        if os.path.exists(file):
            try:
                model_data = joblib.load(file)
                cv_scores = model_data.get('cv_scores', {})
                stock_performance = model_data.get('stock_performance', {})
                
                print(f"\n📊 {name}:")
                if cv_scores:
                    if isinstance(list(cv_scores.values())[0], dict):
                        avg_cv = np.mean([scores['mean'] for scores in cv_scores.values()])
                    else:
                        avg_cv = np.mean([scores.mean() for scores in cv_scores.values()])
                    print(f"  Average CV Accuracy: {avg_cv:.4f}")
                
                if stock_performance:
                    accuracies = [perf['accuracy'] for perf in stock_performance.values()]
                    print(f"  Average Stock Accuracy: {np.mean(accuracies):.4f}")
                    print(f"  Stock Count: {len(stock_performance)}")
                
            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")
        else:
            print(f"\n❌ {name}: File not found")

if __name__ == "__main__":
    # Test expanded model
    test_expanded_model()
    
    # Compare models
    compare_models() 