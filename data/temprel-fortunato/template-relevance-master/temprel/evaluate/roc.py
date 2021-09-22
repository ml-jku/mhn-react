from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def roc_curve(model, test_fps, test_appl_labels, class_idx=0):
    y_score = model.predict(test_fps)
    fpr, tpr, _ = roc_curve(test_appl_labels[:, class_idx].A.reshape(-1), y_score[:, class_idx])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def roc_auc(model, test_fps, test_appl_labels, multi_class='ovr', average='weighted'):
    y_score = model.predict(test_fps)
    mask = test_appl_labels.A.sum(axis=0)>=2
    return roc_auc_score(test_appl_labels[:, mask], y_score[:, mask], multi_class=multi_class, average=average)
