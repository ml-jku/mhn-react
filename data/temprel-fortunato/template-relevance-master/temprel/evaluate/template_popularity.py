import numpy as np

def template_popularity(model, test_fps, train_labels, test_appl_labels, topk=100, batch_size=512):
    class_counts = np.bincount(train_labels)
    pred = model.predict(test_fps)
    ind = np.argsort(-pred)[:, :topk]
    appl_ind = [i[np.isin(i, np.argwhere(appl_label.A))] for i, appl_label in zip(ind, test_appl_labels)]
    appl_pop = np.array([class_counts[i] for i in appl_ind])
    max_length = max([len(a) for a in appl_pop])
    appl_pop = np.vstack([np.pad(pop.astype(float), pad_width=(0, max_length-len(pop)), mode='constant', constant_values=np.NaN) for pop in appl_pop])
    return appl_pop