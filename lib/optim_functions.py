import optuna
from lightgbm import LGBMClassifier
from lib.optim_utils import cross_validation

def objective_lgbm(trial, X_train, y_train, smote_resampling=False, smote_params=None):
    """Objective function for LightGBM hyperparameter optimization using Optuna.
    
    Args:
        trial: Optuna trial object
        X_train: Training feature matrix
        y_train: Training target vector
        smote_resampling (bool, optional): Whether to apply SMOTE resampling. Defaults to False.
        smote_params (dict, optional): Parameters for SMOTE resampling. Defaults to None.
        
    Returns:
        float: PR-AUC-Brier score from cross validation
    """
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 512), 
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000), 
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100), 
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True), 
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True), 
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), 
        'max_depth': trial.suggest_categorical('max_depth', [-1, 3, 10, 20]),  
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    
    clf = LGBMClassifier(**params,
                        n_jobs=-1,
                        verbosity=-1,
                        boosting_type="gbdt",    
                        objective='binary',
                        random_state=42,
                        is_unbalance=True)
    
    pr_auc_brier = cross_validation(X_train, y_train, clf, smote_resampling=smote_resampling, smote_params=smote_params)

    return pr_auc_brier