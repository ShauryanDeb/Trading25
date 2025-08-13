# Model Training Scripts

This directory contains scripts for training, tuning, and optimizing machine learning models for trading strategies.

## Core Files

### Basic Training
- **`train_model.py`** - Basic model training with Random Forest
- **`enhanced_train_model.py`** - Advanced training with options support and multiple model types
- **`multi_stock_train.py`** - Multi-stock model training and ensemble creation

### Model Architectures
- **`advanced_model_architecture.py`** - Complex model architectures and neural networks
- **`enhanced_multi_class_model.py`** - Multi-class classification models
- **`advanced_ensemble_model.py`** - Ensemble model architectures
- **`ensemble_learning.py`** - Ensemble learning strategies and stacking

### Hyperparameter Tuning
- **`tune_model.py`** - Basic hyperparameter tuning
- **`enhanced_hyperparameter_tuning.py`** - Advanced tuning strategies
- **`efficient_hyperparameter_tuning.py`** - Optimized tuning algorithms
- **`advanced_hyperparameter_tuning.py`** - Sophisticated tuning approaches

### Model Optimization
- **`feature_selection_and_regularization.py`** - Feature selection and regularization techniques
- **`quick_overfitting_fix.py`** - Overfitting prevention and model regularization

## Usage Examples

### Basic Model Training
```python
from model_training.train_model import train
model = train(df)
```

### Enhanced Training with Options
```python
from model_training.enhanced_train_model import train_model
model, results = train_model(
    X, y, feature_names,
    model_type="random_forest",
    n_estimators=200
)
```

### Multi-Stock Training
```python
from model_training.multi_stock_train import train_multi_stock_model
ensemble_model = train_multi_stock_model(stock_list)
```

### Hyperparameter Tuning
```python
from model_training.tune_model import tune_hyperparameters
best_params = tune_hyperparameters(X, y)
```

## Model Types Supported

### Classification Models
- Random Forest
- XGBoost
- Gradient Boosting
- Support Vector Machines
- Neural Networks

### Ensemble Methods
- Voting Classifiers
- Stacking
- Bagging
- Boosting

### Multi-Class Models
- One-vs-Rest
- One-vs-One
- Multi-class neural networks

## Training Features

### Data Handling
- Train/test splitting
- Cross-validation
- Stratified sampling
- Time series validation

### Feature Engineering Integration
- Automatic feature selection
- Feature importance analysis
- Dimensionality reduction
- Feature scaling

### Model Evaluation
- Classification metrics
- Confusion matrices
- ROC curves
- Precision-recall curves

### Hyperparameter Optimization
- Grid search
- Random search
- Bayesian optimization
- Genetic algorithms

## Model Persistence

All models are saved with metadata including:
- Model object
- Feature names
- Training parameters
- Performance metrics
- Training date

## Best Practices

1. **Start Simple**: Use `train_model.py` for initial experiments
2. **Add Complexity**: Progress to enhanced versions as needed
3. **Validate Thoroughly**: Use cross-validation and walk-forward analysis
4. **Monitor Overfitting**: Use regularization and early stopping
5. **Ensemble Methods**: Combine multiple models for better performance 