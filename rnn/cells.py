import tensorflow as tf
from ..utils import misc
from . import utils


class GRUCell:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            gru_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(self._units)
                 for _ in range(self._num_layers)]
            )
        else:
            gru_cell = tf.contrib.rnn.GRUCell(self._units)
        return tf.contrib.rnn.OutputProjectionWrapper(gru_cell, nout)


class GRUResidual:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            gru_cell = tf.contrib.rnn.MultiRNNCell(
                [utils.ProjectedResidualWrapper(
                    tf.contrib.rnn.GRUCell(self._units))
                 for _ in range(self._num_layers)]
            )
        else:
            gru_cell = tf.contrib.rnn.GRUCell(self._units)
            gru_cell = utils.ProjectedResidualWrapper(gru_cell)
        return tf.contrib.rnn.OutputProjectionWrapper(gru_cell, nout)


class LSTMCell:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            lstm_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(
                    self._units, state_is_tuple=False)
                 for _ in range(self._num_layers)]
            )
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._units,
                                                     state_is_tuple=False)
        return tf.contrib.rnn.OutputProjectionWrapper(lstm_cell, nout)
