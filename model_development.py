from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np

def tune_models(X_train, y_train, weights_train, scoring='f1_macro', random_state=42):
    """
    Perform hyperparameter tuning for each model.
    
    Parameters:
    scoring : str, default='f1_macro'
        Scoring metric for model selection. Options: 'f1_macro', 'f1_weighted'
    """
    # Define parameter grids for each model
    param_grids = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=random_state, class_weight='balanced'),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'max_iter': [100, 200, 300]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
            'params': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'n_estimators': [100, 200, 300]  
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=random_state),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.5, 0.7, 1.0] 
            }
        }
    }

    tuned_models = {}
    
    for name, config in param_grids.items():
        print(f"\nTuning {name}...")
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train, sample_weight=weights_train)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        tuned_models[name] = grid_search.best_estimator_
    
    return tuned_models

def evaluate_models(X, y, sample_weight=None, random_state=42):
    """
    Evaluate multiple models using cross-validation and optional sample weights.
    
    Parameters:
    X : feature matrix
    y : target vector
    sample_weight : optional array of sample weights
    random_state : random seed for reproducibility
    
    Returns:
    dict : Dictionary containing evaluation results for each model
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y))

    # Split the data with sample weights
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weight, test_size=0.2, random_state=random_state
    )

    # Define and tune our models 
    models = tune_models(X_train, y_train, weights_train, "f1_macro", random_state)
    
    # Store results
    results = {}
    
    # Evaluate each model
    for name, model in models.items():
        # Fit model with sample weights
        model.fit(X_train, y_train, sample_weight=weights_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Store results with weighted metrics
        results[name] = {
            'test_accuracy': accuracy_score(y_test, y_pred, sample_weight=weights_test),
            'test_f1_macro': f1_score(y_test, y_pred, sample_weight=weights_test, average='macro'),
            'classification_report': classification_report(
                y_test, y_pred, 
                sample_weight=weights_test,
                output_dict=True,  
                zero_division=0 
            ),
            'confusion_matrix': confusion_matrix(
                y_test, y_pred, 
                sample_weight=weights_test
            )
        }
    
    return results, X_test, y_test, weights_test, models

def print_results(results):
    """
    Print formatted evaluation results.
    """
    print("\nModel Evaluation Results:")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Test set accuracy: {metrics['test_accuracy']:.3f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("-" * 50)