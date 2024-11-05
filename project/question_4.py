
from question_3 import prediction_module
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
import pandas as pd


def test_scores():
    clf, X_test, y_test, params, selected_features = prediction_module()

    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label='Attrited Customer')
    
    try:
        fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1], pos_label='Attrited Customer')
        auc_roc = roc_auc_score(y_test, y_pred)
    except ValueError as e:
        print(f"Error in ROC computation: {e}")
        fpr, tpr, auc_roc = None, None, None

    dictionary = {
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "accuracy": clf.score(X_test, y_test),
        "f1": f1,
        "params": params,
        "selected_features": selected_features,
        "roc_curve": (fpr, tpr),
        "auc_roc": auc_roc
    }

    return dictionary


# Get test scores
dictionary = test_scores()




