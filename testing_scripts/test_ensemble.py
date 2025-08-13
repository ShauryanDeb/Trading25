import joblib

# Load the advanced ensemble model
model_data = joblib.load('advanced_ensemble_model.pkl')

print("Advanced Ensemble Model Test")
print("=" * 40)
print(f"Model loaded successfully")

# Inspect model structure
print(f"\nModel keys: {list(model_data.keys())}")

# Check if it's the expected structure
if 'ensemble' in model_data:
    ensemble = model_data['ensemble']
    print(f"Ensemble model loaded with {len(ensemble.models)} models")
    print(f"Models: {list(ensemble.models.keys())}")
    
    if hasattr(ensemble, 'selected_features'):
        print(f"Selected features: {len(ensemble.selected_features)}")
        print(f"\nTop 10 selected features:")
        for i, feature in enumerate(ensemble.selected_features[:10]):
            print(f"  {i+1}. {feature}")
    
    if hasattr(ensemble, 'feature_importance'):
        print(f"\nFeature importance available for models: {list(ensemble.feature_importance.keys())}")

if 'results' in model_data:
    results = model_data['results']
    print(f"\nResults keys: {list(results.keys())}")
    if 'accuracy' in results:
        print(f"Ensemble accuracy: {results['accuracy']:.4f}")

if 'model_config' in model_data:
    config = model_data['model_config']
    print(f"\nModel configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}") 