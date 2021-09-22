import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense

class Highway(Layer):
    def __init__(self, activation='relu', transform_activation='sigmoid', **kwargs):
        self.activation = activation
        self.transform_activation = transform_activation
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(units=input_shape[-1], activation=self.activation, bias_initializer='zeros')
        self.dense_gate = Dense(units=input_shape[-1], activation=self.transform_activation, bias_initializer='zeros')
        self.input_dim = input_shape[-1]
        super(Highway, self).build(input_shape)

    def call(self, x):
        transform = self.dense(x)
        transform_gate = self.dense_gate(x)
        carry_gate = K.ones_like(transform_gate) - transform_gate
        output = transform*transform_gate + x*carry_gate
        return output