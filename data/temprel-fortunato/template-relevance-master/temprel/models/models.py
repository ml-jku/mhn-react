import tensorflow as tf
from .layers import Highway
from .metrics import top_k
from .losses import sparse_categorical_crossentropy_from_logits

def build_model(
    input_shape, output_shape, num_hidden, hidden_size, num_highway,
    activation='relu', output_activation=None, dropout=0.0, clipnorm=None,
    optimizer=None, learning_rate=0.001, 
    compile_model=True, loss=None, metrics=None
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    for _ in range(num_hidden):
        model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    for _ in range(num_highway):
        model.add(Highway())
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_shape, activation=output_activation))
    if optimizer is None or optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    if clipnorm is not None:
        optimizer.clipnorm = clipnorm
    if compile_model:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    return model

def relevance(**kwargs):
    loss = sparse_categorical_crossentropy_from_logits
    metrics = [
        top_k(k=1),
        top_k(k=10),
        top_k(k=50),
        top_k(k=100),
    ]
    options = {
        'loss': loss,
        'metrics': metrics
    }
    options.update(kwargs)
    return build_model(**options)

def applicability(**kwargs):
    loss = tf.keras.losses.categorical_crossentropy
    metrics = [
        tf.keras.metrics.Recall(), 
        tf.keras.metrics.Precision()
    ]
    options = {
        'loss': loss,
        'metrics': metrics,
        'output_activation': 'sigmoid'
    }
    options.update(kwargs)
    return build_model(**options)
