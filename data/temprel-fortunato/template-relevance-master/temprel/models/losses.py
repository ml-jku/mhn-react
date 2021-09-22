import tensorflow as tf

def sparse_categorical_crossentropy_from_logits(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)