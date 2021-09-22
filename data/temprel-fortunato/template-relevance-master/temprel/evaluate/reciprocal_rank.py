import numpy as np

def reciprocal_rank(model, test_fps, test_labels):
    pred = model.predict(test_fps)
    ind = np.argsort(-pred)
    ranks = 1 + np.argwhere(test_labels.reshape(-1, 1) == ind)[:, 1]
    return 1/ranks

def reciprocal_rank_by_popularity(model, test_fps, test_labels, train_labels, batch_size=512):
    class_counts = np.bincount(train_labels)
    cls_ind = {}
    for n in range(0, 11):
        cls_ind[n] = np.isin(test_labels, np.argwhere(class_counts==n).reshape(-1))
    cls_ind['med'] = np.isin(test_labels, np.argwhere((class_counts>10) & (class_counts<=50)).reshape(-1))
    cls_ind['high'] = np.isin(test_labels, np.argwhere(class_counts>50).reshape(-1))
    rr = {}
    rr['all'] = reciprocal_rank(model, test_fps, test_labels).tolist()

    for n in range(0, 11):
        rr[n] = reciprocal_rank(model, test_fps[cls_ind[n]], test_labels[cls_ind[n]]).tolist()
    rr['med'] = reciprocal_rank(model, test_fps[cls_ind['med']], test_labels[cls_ind['med']]).tolist()
    rr['high'] = reciprocal_rank(model, test_fps[cls_ind['high']], test_labels[cls_ind['high']]).tolist()
    return rr
