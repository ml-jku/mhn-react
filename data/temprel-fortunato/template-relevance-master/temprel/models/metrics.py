import tensorflow as tf
from functools import partial

def top_k(k=1):
    partial_fn = partial(tf.keras.metrics.sparse_top_k_categorical_accuracy, k=k)
    partial_fn.__name__ = 'top_{}'.format(k)
    return partial_fn