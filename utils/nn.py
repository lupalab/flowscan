import tensorflow as tf
import numpy as np


def lrelu(x, leak=0.2, name='lrelu'):
    """ Assumes leak <= 1.0! """
    return tf.maximum(x, leak*x)


def fc_network(x, outsize, hidden_sizes, reuse=None, output_init_range=None,
               dropout_input=False, dropout_keep_prob=None,
               activation=tf.nn.tanh, name='fc_net'):
    with tf.variable_scope(name, reuse=reuse):
        x_shape = x.get_shape().as_list()
        rank3 = len(x_shape) == 3
        if rank3:
            h = x_shape[2]
            n = tf.shape(x)[1]
            out = tf.reshape(x, [-1, h], name='fc_inp_reshape')
        else:
            assert len(x_shape) == 2
            out = x
        if dropout_input and dropout_keep_prob is not None:
            out = tf.nn.dropout(out, dropout_keep_prob)
        for k, h in enumerate(hidden_sizes):
            with tf.variable_scope('fc_layer_{}'.format(k)):
                h_in = int(out.get_shape()[1])
                W = tf.get_variable('W', shape=(h_in, h), dtype=tf.float32)
                b = tf.get_variable('b', shape=(h, ), dtype=tf.float32)
                out = activation(tf.nn.xw_plus_b(out, W, b, 'linear'))
                if dropout_keep_prob is not None:
                    out = tf.nn.dropout(out, dropout_keep_prob)
        if output_init_range is not None:
            initializer = tf.random_uniform_initializer(
                minval=-output_init_range, maxval=output_init_range
            )
        else:
            initializer = None
        with tf.variable_scope('linear', initializer=initializer):
            h_in = int(out.get_shape()[1])
            out = tf.nn.xw_plus_b(
                out,
                tf.get_variable('W', shape=(h_in, outsize), dtype=tf.float32),
                tf.get_variable('b', shape=(outsize, ), dtype=tf.float32),
                'projection'
            )
        if rank3:
            out = tf.reshape(out, [-1, n, outsize], name='fc_out_reshape')
    return out


def batch_norm(x, train=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(
        x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
        is_training=train, scope=name
    )


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           train=True, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev),
            trainable=train
        )
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1],
                            padding='SAME')
        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0),
                                 trainable=train)
        shape = conv.get_shape().as_list()
        shape[0] = -1
        conv = tf.reshape(tf.nn.bias_add(conv, biases), shape)
    return conv


def convnet(inputs, ncode, outdims=[64, 64, 64], khw=5, shw=2,
            reuse=None, train=True, name='convnet', return_hidden=False,
            output_irange=1e-6, do_batch_norm=False):
    """Convnet using default 5x5 convolutions w/ stride 2.
    Args:
        inputs: n x h x w x c tensor, images
        ncode: int, number of dimensions for codes
        outdims: list of ints, number of channels in each convolution
        khw: int, size of convolution kernel
        shw: int, stride for convolution
        reuse: boolean, reuse variables?
        train: boolean, train phase for batch_normalizer
        name: string, variable scope
    Returns:
        code: n x ncode tensor
    """
    inps = inputs
    hidden = []
    with tf.variable_scope(name, reuse=reuse):
        for k, dims in enumerate(outdims):
            if k == 0 and not do_batch_norm:
                inps = lrelu(
                    conv2d(inps, dims, k_h=khw, k_w=khw, d_h=shw, d_w=shw,
                           name='conv{}'.format(k+1)))
            else:
                inps = lrelu(batch_norm(
                    conv2d(inps, dims, k_h=khw, k_w=khw, d_h=shw, d_w=shw,
                           name='conv{}'.format(k+1)),
                    name='bn{}'.format(k), train=train))
            hidden.append(inps)
        ten_dims = np.prod(inps.get_shape().as_list()[1:])
        if ncode is None:
            code = tf.reshape(inps, [-1, ten_dims])
        else:
            code = fc_network(tf.reshape(inps, [-1, ten_dims]), ncode, [],
                              output_init_range=output_irange)
    print([inputs]+hidden+[code])
    if return_hidden:
        return code, hidden
    return code


def get_default_condition_params(input_type):
    if input_type is 'image':
        cond_net = 'conv'
        cond_dict = {
            'ncode': None,
            'outdims': [64, 64, 64],
            'khw': 5,
            'shw': 2,
            'reuse': None,
            'train': True,
            'return_hidden': False,
            'output_irange': 1e-6,
            'do_batch_norm': False
        }
    elif input_type is 'vector':
        cond_net = 'fc'
        cond_dict = {
            'outsize': 64,
            'hidden_sizes': [64, 128],
            'reuse': None,
            'output_init_range': None,
            'dropout_input': False,
            'dropout_keep_prob': None,
            'activation': tf.nn.tanh
        }
    else:
        raise TypeError("input_type is {} but must be one of "
                      + "'image' or 'vector'".format(input_type))

    return cond_net, cond_dict


def append_default_condition_params(input_type, cond_dict):
    def_dict = get_default_condition_params(input_type)
    for kk in def_dict.keys():
        if kk not in cond_dict:
            cond_dict[kk] = def_dict[kk]

    return cond_dict
