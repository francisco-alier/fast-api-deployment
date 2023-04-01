from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
import numpy as np
import logging
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Configure the logging module
    logging.basicConfig(
        filename='model_training_log.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
                    'n_estimators': [20, 50, 80],
                    'max_depth': [2, 4, 6],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'subsample': [0.8, 1.0],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 3],
                    'random_state': [42]
                    }
    # Define the GBM classifier
    clf = GradientBoostingClassifier()

    # Define the grid search with cross-validation
    grid_search = GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1
    )


    grid_search.fit(X_train, y_train)
        
    # Extract the best hyperparameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best hyperparameters and the best score
    print(f'Best parameters: {best_params}')
    print(f'Best score: {best_score}')
    
    logging.info(f'Best parameters: {best_params}')
    logging.info(f'Mean CV score: {best_score:.3f}')

    # Train the classifier with the best hyperparameters
    clf = GradientBoostingClassifier(**best_params)
    clf.fit(X_train, y_train)

    return clf
    

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds

def compute_metrics_on_slices(X, y, feature_name, model):
    """ Computes metrics on model slices given a specific categorical feature

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    feature : string
        Feature to be analyzed - only works with categorical data
    model:´
        Trained Classifier
    Returns
    -------
    preds : pandas dataframe
        Performance metrics for each value of the fixed feature.
    """

    # Get the unique values of the fixed feature
    fixed_values = list(X[feature_name].unique())

     # Create an empty DataFrame to store the performance metrics
    metrics_df = pd.DataFrame(columns=['feature_value', 'precision', 'recall', 'fbeta'])
    
    # Compute the performance metrics for each value of the fixed feature
    for value in fixed_values:
        # Select the data that has the fixed feature value
        mask = X[:, feature_name] == value
        X_fixed = X[mask]
        y_fixed = y[mask]
        
        # Predict the target values
        preds = inference(model, X_fixed)
        
        # Compute the performance metrics
        precision, recall, fbeta = compute_model_metrics(y_fixed, preds)
        
        # Add the performance metrics to the DataFrame
        row = {
            'feature_value': value,
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta
        }

        metrics_df = metrics_df.append(row, ignore_index=True)

        #Export the metrics to a .txt file
        with open('slice_output.txt', 'w') as f:
            f.write(metrics_df.to_string(index=False))
        
    return metrics_df

