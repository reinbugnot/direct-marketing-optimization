import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss,average_precision_score
from imblearn.over_sampling import SMOTE

def save_params(params, fname):
    """Save model parameters to a JSON file.
    
    Args:
        params (dict): Dictionary of model parameters to save
        fname (str): Name of file to save parameters to (without .json extension)
    """
    os.makedirs('train_params', exist_ok=True)

    with open(f'train_params/{fname}.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)

def load_params(fname, print_output=False):
    """Load model parameters from a JSON file.
    
    Args:
        fname (str): Path to JSON file containing parameters
        print_output (bool, optional): Whether to print the loaded parameters. Defaults to False.
        
    Returns:
        dict: Dictionary of model parameters loaded from file
    """
    with open(fname, 'r') as json_file:
        params = json.load(json_file)
        
    if print_output:
        print(params)
        
    return params

def cross_validation(X, y, model, cv=5, print_status=False, smote_resampling=False, smote_params=None):
    """Perform stratified k-fold cross validation with optional SMOTE resampling.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        model: Sklearn-compatible model object with fit/predict_proba methods
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        print_status (bool, optional): Whether to print progress updates. Defaults to False.
        smote_resampling (bool, optional): Whether to apply SMOTE resampling. Defaults to False.
        smote_params (dict, optional): Parameters for SMOTE resampling. Defaults to None.
        
    Returns:
        float: Mean PR-AUC-Brier score across folds
    """
    kfold = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 42)
    
    scores = []
    for j, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

        # Apply SMOTE if toggled on
        if smote_resampling:
            smote_params = smote_params or {}
            smote = SMOTE(**smote_params, random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            X_train = X_train_resampled
            y_train = y_train_resampled
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_valid)[:,1]
        
        # Calculate PR-AUC-Brier score
        pr_auc = average_precision_score(y_valid, y_pred_proba)
        brier_score = brier_score_loss(y_valid, y_pred_proba)
        pr_auc_brier = pr_auc - (brier_score * 0.1)

        scores.append(pr_auc_brier)
        
        if print_status:
            print(f'Fold {j+1} Done. PR-AUC-Brier: {pr_auc_brier}')
        
    if print_status:
        print(f'\nCompleted {cv}-fold CV. Mean PR-AUC-Brier: {np.mean(scores)}')
        
    return np.mean(scores)