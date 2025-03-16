from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, brier_score_loss, recall_score, precision_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def train_classifier(clf, X, y, print_metrics=False, val_size=0.2, smote_resampling=False, smote_params=None):
    """Train a classifier with optional SMOTE resampling and validation metrics.
    
    Args:
        clf: Sklearn-compatible classifier with fit/predict/predict_proba methods
        X: Feature matrix
        y: Target vector
        print_metrics (bool, optional): Whether to print validation metrics. Defaults to False.
        val_size (float, optional): Validation set size as fraction. Defaults to 0.2.
        smote_resampling (bool, optional): Whether to apply SMOTE resampling. Defaults to False.
        smote_params (dict, optional): Parameters for SMOTE resampling. Defaults to None.
        
    Returns:
        clf: Trained classifier model
    """
    # Train-Test Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
    
    # Apply SMOTE for resampling to upsample minority class in the training set
    if smote_resampling:
        smote_params = smote_params or {}
        smote = SMOTE(**smote_params, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    clf.fit(X_train, y_train)

    # Validation
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)[:,1]

    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    brier_score = brier_score_loss(y_val, y_pred_proba)
    pr_auc_brier = pr_auc - (brier_score * 0.1)
    
    if print_metrics:
        print("Model Performance on Validation Set:")
        print("-" * 25) 
        print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
        print(f"Precision Score: {precision:.4f}")
        print(f"Recall Score: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"PR-AUC-Brier Score: {pr_auc_brier:.4f}")
        print("=" * 50)
        print("\n")
    
    return clf

def evaluate_model_performance(clf, X_test, y_test, print_metrics=True):
    """Evaluate classifier performance on test data.
    
    Args:
        clf: Trained classifier model
        X_test: Test feature matrix
        y_test: Test target vector
        print_metrics (bool, optional): Whether to print test metrics. Defaults to True.
        
    Returns:
        float: PR-AUC-Brier score on test set
    """
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:,1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier_score = brier_score_loss(y_test, y_pred_proba)
    pr_auc_brier = pr_auc - (brier_score * 0.1)

    if print_metrics:
        print("Model Performance on Test Set:")
        print("-" * 25) 
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(f"Precision Score: {precision:.4f}")
        print(f"Recall Score: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"PR-AUC-Brier Score: {pr_auc_brier:.4f}")
        print("=" * 50)
        print("\n")

    return pr_auc_brier