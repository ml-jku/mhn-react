import numpy as np

def appl_recall_and_precision(model, test_fps, test_appl_labels, k=10):
    row, col = test_appl_labels.nonzero()
    scores = model.predict(test_fps)
    indices = np.argsort(-scores, axis=1)
    appl_recall = []
    appl_precision = []
    for n in range(len(indices)):
        n_appl = np.isin(indices[n][:k], col[row==n]).sum()
        total_appl = col[row==n].size
        total_reccommended = k
        appl_recall.append(n_appl/total_appl)
        appl_precision.append(n_appl/total_reccommended)
    return appl_recall, appl_precision

def topk_appl_recall_and_precision(model, test_fps, test_appl_labels, k=[1, 5, 10, 25, 50, 100, 250, 1000]):
    recall = {}
    precision = {}
    for kval in k:
        r, p = appl_recall_and_precision(model, test_fps, test_appl_labels, k=kval)
        recall[kval] = r
        precision[kval] = p
    return recall, precision
