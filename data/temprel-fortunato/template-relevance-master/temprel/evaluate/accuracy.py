import numpy as np

def accuracy_by_popularity(model, test_fps, test_labels, train_labels, batch_size=512):
    class_counts = np.bincount(train_labels)
    cls_ind = {}
    for n in range(0, 11):
        cls_ind[n] = np.isin(test_labels, np.argwhere(class_counts==n).reshape(-1))
    cls_ind['med'] = np.isin(test_labels, np.argwhere((class_counts>10) & (class_counts<=50)).reshape(-1))
    cls_ind['high'] = np.isin(test_labels, np.argwhere(class_counts>50).reshape(-1))
    performance = {}
    performance['all'] = model.evaluate(test_fps, test_labels, batch_size=batch_size)
    for n in range(0, 11):
        if len(test_fps[cls_ind[n]]) == 0:
            continue
        performance[n] = model.evaluate(test_fps[cls_ind[n]], test_labels[cls_ind[n]], batch_size=batch_size)
    performance['med'] = model.evaluate(test_fps[cls_ind['med']], test_labels[cls_ind['med']], batch_size=batch_size)
    performance['high'] = model.evaluate(test_fps[cls_ind['high']], test_labels[cls_ind['high']], batch_size=batch_size)
    return performance
