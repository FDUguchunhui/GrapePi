from torch_geometric.graphgym.register import register_metric

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

@register_metric('accuracy')
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
accuracy_score

@register_metric('AUC')
def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

@register_metric('f1')
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)
