import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN

from layers import ReflectionPadding2D


class ConvGate(Layer):
    def __init__(self, activation='sigmoid', **kwargs):
        super().__init__(**kwargs)
        self.padding = ReflectionPadding2D(padding=(1, 1))
        self.conv = Conv2D(256, kernel_size=(3, 3))
        self.activation = Activation(activation) if activation is not None else None

    def call(self, x):
        x = self.padding(x)
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class PaddingGate(ConvGate):
    def __init__(self, activation='sigmoid', **kwargs):
        super().__init__(activation, **kwargs)
        # self.conv = SNConv2D(256, kernel_size=(3, 3), kernel_initializer='he_uniform')


class ConvGRUCell(Layer):
    def __init__(self, input_size, update_gate, reset_gate, output_gate, **kwargs):
        self.update_gate = update_gate
        self.reset_gate = reset_gate
        self.output_gate = output_gate
        self.state_size = input_size
        super(ConvGRUCell, self).__init__(**kwargs)

    def call(self, inputs, states):
        x = inputs
        state, = states
        xh = tf.concat((x, state), axis=-1)
        z = self.update_gate(xh)
        reset = self.reset_gate(xh)
        out = self.output_gate(tf.concat((x, reset * state), axis=-1))
        new_state = z * state + (1 - z) * tf.math.tanh(out)
        return out, [new_state]


def ConvGRU(state_size, update_gate, reset_gate, output_gate, return_sequences=False, **kwargs):
    cell = ConvGRUCell(state_size, update_gate, reset_gate, output_gate, **kwargs)
    return RNN(cell, return_sequences=return_sequences, **kwargs)


class CustomGateGRU(Layer):
    def __init__(self, 
        update_gate=None, reset_gate=None, output_gate=None,
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = update_gate
        self.reset_gate = reset_gate
        self.output_gate = output_gate
        # self.return_sequences = return_sequences
        self.time_steps = time_steps

    def call(self, inputs, ):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            # if self.return_sequences:
            h_all.append(h)
        return tf.stack(h_all,axis=1)  # if self.return_sequences else h

    def get_config(self):
        return super(CustomGateGRU, self).get_config()
