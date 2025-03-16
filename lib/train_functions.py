from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def train_classifier(clf, X, y, print_metrics=True, val_size=0.2, smote_resampling=False, smote_params=None):

    # Train-Test Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
    
    # Apply SMOTE for resampling to upsample minority class in the training set
    if smote_resampling:
        smote_params = smote_params or {}
        smote = SMOTE(**smote_params)
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
    
    if print_metrics:
        print("Model Performance on Validation Set:")
        print("-" * 25) 
        print(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
        print(f"Precision Score: {precision:.4f}")
        print(f"Recall Score: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("=" * 50)
        print("\n")
    
    return clf, (precision, recall, f1, pr_auc, roc_auc)

# def evaluate_model_performance(clf, X_test, y_test, print_metrics=True):

#     y_pred = clf.predict(X_test)
#     y_pred_proba = clf.predict_proba(X_test)[:,1]

#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     pr_auc = average_precision_score(y_test, y_pred_proba)
#     roc_auc = roc_auc_score(y_test, y_pred_proba)

#     if print_metrics:
#         print("Model Performance on Test Set:")
#         print("-" * 25)
#         print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
#         print(f"Precision Score: {precision:.4f}")
#         print(f"Recall Score: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"PR-AUC Score: {pr_auc:.4f}")
#         print(f"ROC-AUC Score: {roc_auc:.4f}")
#         print("=" * 50)

#     return y_pred, y_pred_proba