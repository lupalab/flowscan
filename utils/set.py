import tensorflow as tf
from . import nn


def embeding_network(inputs, embed_layers=256, embed_size=128):
    with tf.variable_scope('embed'):
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        d = int(inputs.get_shape()[-1])
        points = tf.reshape(inputs, [N*n, d])

        # weights
        W0 = tf.get_variable(
            'W0', shape=(d, embed_layers), dtype=tf.float32)
        b0 = tf.get_variable(
            'b0', shape=(embed_layers, ), dtype=tf.float32)
        W1 = tf.get_variable(
            'W1', shape=(embed_layers, embed_layers), dtype=tf.float32)
        b1 = tf.get_variable(
            'b1', shape=(embed_layers, ), dtype=tf.float32)
        W2 = tf.get_variable(
            'W2', shape=(embed_layers, embed_layers), dtype=tf.float32)
        b2 = tf.get_variable(
            'b2', shape=(embed_layers, ), dtype=tf.float32)
        W3 = tf.get_variable(
            'W3', shape=(embed_layers, embed_size), dtype=tf.float32)
        b3 = tf.get_variable(
            'b3', shape=(embed_size, ), dtype=tf.float32)

        # forward
        # Standard Layer0
        # [N*n, embed_layers]
        y = tf.nn.elu(tf.nn.xw_plus_b(points, W0, b0, 'linear0'))
        # PermEqui Layer1
        # [N, n, embed_layers]
        y = tf.reshape(y, [N, n, embed_layers])
        # [N, 1, embed_layers]
        ym = tf.reduce_max(y, 1, True)
        # [N, n. embed_layers]
        y = y - ym
        y = tf.reshape(y, [N*n, embed_layers])
        # PermEqui Layer2
        y = tf.nn.elu(tf.nn.xw_plus_b(y, W1, b1, name='linear1'))
        y = tf.reshape(y, [N, n, embed_layers])
        ym = tf.reduce_max(y, 1, True)
        y = y - ym
        y = tf.reshape(y, [N*n, embed_layers])
        y = tf.nn.elu(tf.nn.xw_plus_b(y, W2, b2, name='linear2'))
        # Standard Layer3
        point_feats = tf.nn.xw_plus_b(y, W3, b3, 'linear3')
        point_feats_tens = tf.reshape(point_feats, [N, n, embed_size])
        # Set features
        embed_feats = tf.concat((tf.reduce_mean(point_feats_tens, 1),
                                 tf.reduce_max(point_feats_tens, 1)), -1)

        return embed_feats


def permequi(inputs, embed_size, hidden_sizes=[],
             reduction='max', combination='minus',
             activation=tf.nn.tanh, pool_point_feats=True):
    """ General permutation equivariant layer on sets X = {x_1, ..., x_n}.
    Maps X to {f(x_1), ..., f(x_n)} and {g(x_1), ..., g(x_n)} (w/ g == f by
    defualt). Outputs set
        {combine(f(x_j), reduce_k( g(x_k) ))}_{j=1}^n  (*)
    observed as matrix, and where combine is an operation specified by the
    combination paramter.  E.g. 'minus', where combine(x, y) = x - y.
    Args:
        inputs: N x n x d real tensor. Inputs[i, :, :] is ith set,
                possibly padded to reach n elements.
        embed_size: int. Number of dims for output of point mappings f, g.
        hidden_size: hidden unit sizes for f, g maps.
        reduction: 'max', 'sum, or 'mean', type of reduction to do on sets.
        combination: 'minus', 'times', 'concat', type of combine function.
        activation: function. Type of hidden unit to use for f and g point
            mappings.
        pool_point_feats: boolean. If true then f = g.
    Returns:
        y: N x n x embed_size permutation equivariant mapping (*) or (**)
           concated into matrices. If combination == 'concat' last tensor dim
           is 2*embed_size.
    """
    # Evaluate fcnet on each point
    with tf.variable_scope('point_feats'):
        y = nn.fc_network(inputs, embed_size, hidden_sizes,
                          activation=activation)

    # Permutation Equivariant
    if reduction is not None:
        # Use different network for set feats?
        if pool_point_feats:
            set_feats = y
        else:
            with tf.variable_scope('set_feats'):
                set_feats = nn.fc_network(inputs, embed_size, hidden_sizes,
                                          activation=activation)

        # Reduce to get set features
        if reduction == 'mean':
            ym = tf.reduce_mean(set_feats, 1, keepdims=True)
        elif reduction == 'sum':
            ym = tf.reduce_sum(set_feats, 1, keepdims=True)
        else:
            ym = tf.reduce_max(set_feats, 1, keepdims=True)

        # Combine set features with each point
        if combination == 'minus':
            y -= ym  # [N, n. embed_size]
        elif combination == 'times':
            y *= ym  # [N, n. embed_size]
        else:  # concat
            y = repeat_set_features(y, ym)  # [N, n. 2*embed_size]
    return y


def repeat_set_features(inputs, set_feats, name='rep_sfeat'):
    """ Replicate and concatenate set feats along points' dimension.
    Args:
        inputs: N x n x d real tensor.
        set_features: N x p or N x 1 x p real tensor.
    Returns:
        cat_feats: N x n x (d+p) real tensor.
    """
    with tf.variable_scope(name):
        rank = len(set_feats.get_shape().as_list())
        if rank == 2:
            set_feats_tensor = tf.expand_dims(set_feats, 1)
        else:
            set_feats_tensor = set_feats
        n = tf.shape(inputs)[1]
        set_feats_tensor = tf.tile(set_feats_tensor, [1, n, 1])
        cat_feats = tf.concat([inputs, set_feats_tensor], 2)
    return cat_feats


def deepset(inputs, embed_size, hidden_sizes=[],
            reduction='max', combination='minus',
            activation=tf.nn.elu, pool_point_feats=True,
            concat_max=True):
    # Set permequi
    tens = inputs
    with tf.variable_scope('permequiv1'):
        tens = permequi(
            tens, embed_size, reduction=reduction,
            hidden_sizes=hidden_sizes, activation=activation,
            combination=combination)
    with tf.variable_scope('permequiv2'):
        tens = permequi(
            tens, embed_size, reduction=reduction,
            hidden_sizes=hidden_sizes, activation=activation,
            combination=combination)
    set_feats = tf.reduce_mean(tens, 1, keepdims=False)
    if concat_max:
        set_feats = tf.concat(
            (set_feats, tf.reduce_max(tens, 1, keepdims=False)), -1)
    return set_feats
