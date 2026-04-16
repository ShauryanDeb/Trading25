# Evaluation Scripts

This directory contains scripts for model evaluation, performance metrics calculation, and walk-forward analysis.

## Core Files

### Basic Evaluation
- **`evaluate_model.py`** - Basic model evaluation and metrics calculation
- **`enhanced_evaluate_model.py`** - Comprehensive model evaluation with detailed analysis

### Ensemble Evaluation
- **`test_ensemble_performance.py`** - Ensemble model testing and performance comparison

### Walk-Forward Analysis
- **`walk_forward_testing.py`** - Walk-forward analysis for time series validation
- **`walk_forward_expanded.py`** - Extended walk-forward testing with multiple scenarios

## Usage Examples

### Basic Model Evaluation
```python
from evaluation.evaluate_model import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
```

### Enhanced Evaluation
```python
from evaluation.enhanced_evaluate_model import comprehensive_evaluation
results = comprehensive_evaluation(
    model, X_test, y_test,
    feature_names=feature_names,
    include_plots=True
)
```

### Ensemble Performance Testing
```python
from evaluation.test_ensemble_performance import test_ensemble
ensemble_results = test_ensemble(ensemble_model, test_data)
```

### Walk-Forward Analysis
```python
from evaluation.walk_forward_testing import walk_forward_analysis
wf_results = walk_forward_analysis(
    model, data, 
    window_size=252,  # 1 year
    step_size=21      # 1 month
)
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision-Recall AUC**: Area under the precision-recall curve

### Trading-Specific Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return / Maximum drawdown

### Model Performance Metrics
- **Feature Importance**: Relative importance of features
- **Confusion Matrix**: Detailed classification results
- **Classification Report**: Comprehensive classification metrics
- **Cross-Validation Scores**: Robust performance estimates

## Walk-Forward Analysis

### Purpose
- Validate model performance over time
- Detect model decay and concept drift
- Ensure out-of-sample performance
- Test strategy robustness

### Parameters
- **Window Size**: Training period length
- **Step Size**: Forward movement increment
- **Minimum Samples**: Minimum required for training
- **Performance Threshold**: Minimum acceptable performance

### Output
- Performance over time
- Model stability metrics
- Concept drift detection
- Optimal retraining frequency

## Evaluation Features

### Visualization
- ROC curves
- Precision-recall curves
- Feature importance plots
- Performance over time charts
- Confusion matrix heatmaps

### Statistical Analysis
- Confidence intervals
- Statistical significance testing
- Performance distribution analysis
- Correlation analysis

### Model Comparison
- Multiple model comparison
- Statistical significance testing
- Performance ranking
- Ensemble vs. individual model analysis

## Best Practices

1. **Use Multiple Metrics**: Don't rely on a single metric
2. **Cross-Validation**: Use time series cross-validation
3. **Out-of-Sample Testing**: Always test on unseen data
4. **Walk-Forward Analysis**: Validate performance over time
5. **Statistical Significance**: Test if improvements are significant
6. **Feature Analysis**: Understand what drives predictions
7. **Model Interpretability**: Ensure models are explainable

## Output Formats

### Standard Metrics
```python
{
    'accuracy': 0.75,
    'precision': 0.72,
    'recall': 0.78,
    'f1_score': 0.75,
    'roc_auc': 0.82
}
```

### Detailed Analysis
```python
{
    'metrics': {...},
    'feature_importance': {...},
    'confusion_matrix': [...],
    'classification_report': '...',
    'plots': {...}
}
```

### Walk-Forward Results
```python
{
    'periods': [...],
    'performance': [...],
    'stability_metrics': {...},
    'recommendations': [...]
}
``` 