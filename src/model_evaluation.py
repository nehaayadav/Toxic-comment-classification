import numpy as np
import pandas as pd

from sklearn.metrics import (                         # Model evaluation metrics
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from IPython.display import display                   # Better display of DataFrames in notebooks


def evaluate_model(y_true, y_pred, y_proba, labels):

    results = {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    # Global metrics
    for avg in ["micro","macro"]:
        results[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        results[f'recall_{avg}']    = recall_score(y_true, y_pred, average=avg, zero_division=0)
        results[f'f1_{avg}']        = f1_score(y_true, y_pred, average=avg, zero_division=0)

    results['roc_auc_macro'] = roc_auc_score(y_true, y_proba, average="macro")
    results['roc_auc_micro'] = roc_auc_score(y_true, y_proba, average="micro")

    print("\n--- Global Metrics ---")
    for k,v in results.items():
        print(f"{k}: {v:.4f}")

    rows = []
    for i,label in enumerate(labels):

        prec = precision_score(y_true[:,i], y_pred[:,i], zero_division=0)
        rec  = recall_score(y_true[:,i], y_pred[:,i], zero_division=0)
        f1   = f1_score(y_true[:,i], y_pred[:,i], zero_division=0)

        auc = None
        if len(set(y_true[:,i])) > 1:
            auc = roc_auc_score(y_true[:,i], y_proba[:,i])

        rows.append([label,prec,rec,f1,auc])

    per_label_df = pd.DataFrame(rows, columns=["Label","Precision","Recall","F1","ROC-AUC"])

    print("\n--- Per-label Metrics ---")
    display(per_label_df)

    return results, per_label_df