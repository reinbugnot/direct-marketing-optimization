import json
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from imblearn.over_sampling import SMOTE

def save_params(params, fname):

    os.makedirs('train_params', exist_ok=True)

    with open(f'train_params/{fname}.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)


# Code to load parameters from JSON
def load_params(fname, print_output=False):
    with open(fname, 'r') as json_file:
        params = json.load(json_file)
        
    if print_output:
        print(params)
        
    return params

def cross_validation(X, y, model, cv=5, print_status=False, smote_resampling=False, smote_params=None):
    
    kfold = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 42)
    
    scores = []
    for j, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]

        # Apply SMOTE for resampling to upsample minority class in the training set
        if smote_resampling:
            smote_params = smote_params or {}
            smote = SMOTE(**smote_params)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_valid)[:,1]
        
        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(y_valid, y_pred_proba)

        # Calculate Brier Score (probability calibration metric)
        brier_score = brier_score_loss(y_valid, y_pred_proba)

        # Calculate Optimization Score
        roc_auc_brier = roc_auc - (brier_score * 0.1)
        
        # Append calculated optimization score to scores list
        scores.append(roc_auc_brier)
        
        if print_status:
            print(f'Fold {j} Done. AUC: {roc_auc}, Brier Score: {brier_score}, Final Optim Score: {roc_auc_brier}')
        
    if print_status:
        print(f'\nCompleted {cv}-fold CV. Mean AUC: {np.mean(scores)}')
        
    return np.mean(scores)