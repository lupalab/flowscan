""" Transformation of variable component of TANs.
- Transformations are function that
  - take in:
    - an input `[N x d]`
    - (and possibly) a conditioning value `[N x p]`
  - return:
    - transformed covariates `[N x d]`
    - log determinant of the Jacobian `[N]` or scalar
    - inverse mapping `[function: N x d (, N x p) -> N x d]`
- `transformer` takes in a list of transformations and
  composes them into single transformation.
"""

import tensorflow as tf
import numpy as np
import scipy.linalg as linalg # noqa
from ..utils import nn
from ..utils import set as us


# %% Transformation composition
#
def transformer(inputs, transformations, conditioning=None, verbose=False,
                forward_tensors=None):
    """Makes transormation on the r.v. X
    Args:
        inputs: N x d tensor of inputs
        transformations: list of functions that take input (and conditioning)
            variables to transform and return output, logdet of Jacobian,
            and inverse for transformation.
        conditioning: N x p tensor of conditioning values
    Returns:
        y: N x d tensor of transformed values
        logdet: scalar tensor with the log determinant corresponding to
            the transformation.
        invmap: function that takes in N x d tensor of the transformed r.v.s
            and outputs the r.v. in originals space.
    """
    # Apply transformations.
    y = inputs
    invmaps = []
    logdet = 0.0
    for i, trans in enumerate(transformations):
        with tf.variable_scope('transformation_{}'.format(i)):
            try:
                y, ldet, imap = trans(y, conditioning)
            # except TypeError as terr:
            #    fprint(terr)
            except TypeError as terr:  # Does not take in conditioning values.
                y, ldet, imap = trans(y)
            if forward_tensors is not None:
                forward_tensors.append(y)
            logdet += ldet
            invmaps.append(imap)

    # Make inverse by stacking inverses in reverse order.
    ntrans = len(invmaps)
    if verbose:
        print(invmaps[::-1])

    def invmap(z, conditioning=None):
        for i in range(ntrans-1, -1, -1):
            with tf.variable_scope('transformation_{}'.format(i)):
                try:
                    z = invmaps[i](z, conditioning)
                except TypeError:  # Does not take in conditioning values.
                    z = invmaps[i](z)
        return z
    return y, logdet, invmap


# %% Simple Transformations.
#
def rescale(x, init_constant=None, name='rescale'):
    """Rescale z = s*x."""
    with tf.variable_scope(name) as scope:
        d = int(x.get_shape()[1])
        if init_constant is not None:
            s = tf.get_variable(
                's', initializer=init_constant*tf.ones((1, d)),
                dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
        y = tf.multiply(x, s, name='y')
        logdet = tf.reduce_sum(tf.log(tf.abs(s)))

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            # TODO: neccesarryryryry?
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            x = tf.divide(y, s, name='y_inv')
            return x

    return y, logdet, invmap


def log_rescale(x, init_zeros=True, name='rescale'):
    """Rescale z = exp(s)*x"""
    with tf.variable_scope(name) as scope:
        d = int(x.get_shape()[1])
        if init_zeros:
            s = tf.get_variable(
                's', initializer=tf.zeros((1, d)), dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
        y = tf.multiply(x, tf.exp(s), name='y')
        logdet = tf.reduce_sum(s)

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            # TODO: neccesaary?
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            x = tf.divide(y, tf.exp(s), name='y_inv')
            return x

    return y, logdet, invmap


def shift(x, init_zeros=True, name='shift'):
    """Shift z = x + b."""
    with tf.variable_scope(name) as scope:
        d = int(x.get_shape()[1])
        if init_zeros:
            s = tf.get_variable(
                's', initializer=tf.zeros((1, d)), dtype=tf.float32)
        else:
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
        y = x + s
        logdet = tf.zeros([1])

        # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            # TODO: neccesaary?
            s = tf.get_variable('s', shape=(1, d), dtype=tf.float32)
            x = y - s
            return x

    return y, logdet, invmap


def negate(x, name='negate'):
    """Negate z = -x."""
    with tf.variable_scope(name) as scope:
        y = -x
        logdet = 0.0

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            return -y

    return y, logdet, invmap


def logit_transform(x, alpha=0.05, max_val=256.0, name='logit_transform',
                    logdet_mult=None):
    """Logit transform for compact values."""
    # print('Using logit transform')

    def logit(x):
        return tf.log(x) - tf.log(1.0-x)

    with tf.variable_scope(name) as scope:
        sig = alpha + (1.0-alpha)*x/max_val
        z = logit(sig)
        logdet = tf.reduce_sum(
            tf.log(1-alpha)-tf.log(sig)-tf.log(1.0-sig)-tf.log(max_val), 1)
        if logdet_mult is not None:
            logdet = logdet_mult*logdet

    # inverse
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            arg = 1.0/(1.0 + tf.exp(-z))
            return (arg-alpha)*max_val/(1.0-alpha)
    return z, logdet, invmap


# %% Permutation functions.
#
def reverse(x, name='reverse'):
    """Reverse along last dimension."""
    with tf.variable_scope(name) as scope:
        z = tf.reverse(x, [-1])
        logdet = 0.0

    # Inverse map
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            x = tf.reverse(z, [-1])
            return x
    return z, logdet, invmap


def permute(x, perm, name='perm'):
    """Permutes according perm along last dimension."""
    with tf.variable_scope(name) as scope:
        z = tf.transpose(tf.gather(tf.transpose(x), perm))
        logdet = tf.zeros([1])

    # Inverse map
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            x = tf.transpose(tf.gather(tf.transpose(z), invperm(perm)))
            return x
    return z, logdet, invmap


def invperm(perm):
    """Returns the inverse permutation."""
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


# %% Linear mapping functions.
#
def get_LU_map(mat_params, b):
    """Make the matrix for linear map y^t = x^t (L U) + b^t.
    Args:
        mat_params: d x d array of matrix parameters. Contains lower and upper
            matrices L, U. L has unit diagonal.
        b: d length array of biases
    Returns:
        A: the linear map matrix resulting from the multiplication of L and U.
        logdet: the log determinant of the Jacobian for this transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope('LU'):
        with tf.variable_scope('unpack'):
            # Unpack the mat_params and U matrices
            d = int(mat_params.get_shape()[0])
            U = tf.matrix_band_part(mat_params, 0, -1)
            L = tf.eye(d) + mat_params*tf.constant(
                np.tril(np.ones((d, d), dtype=np.float32), -1),
                dtype=tf.float32, name='tril'
            )
            A = tf.matmul(L, U, name='A')
        with tf.variable_scope('logdet'):
            # Get the log absolute determinate
            logdet = tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(U))))

        # Inverse map
        def invmap(y):
            with tf.variable_scope('invmap'):
                Ut = tf.transpose(U)
                Lt = tf.transpose(L)
                yt = tf.transpose(y)
                sol = tf.matrix_triangular_solve(Ut, yt-tf.expand_dims(b, -1))
                x = tf.transpose(
                    tf.matrix_triangular_solve(Lt, sol, lower=False)
                )
                return x
    return A, logdet, invmap


def linear_map(x, init_mat_params=None, init_b=None, mat_func=get_LU_map,
               trainable_A=True, trainable_b=True, irange=1e-10,
               name='linear_map'):
    """Return the linearly transformed, y^t = x^t * mat_func(mat_params) + b^t,
    log determinant of Jacobian and inverse map.
    Args:
        x: N x d real tensor of covariates to be linearly transformed.
        init_mat_params: tensor of parameters for linear map returned by
            mat_func(init_mat_params, b) (see get_LU_map above).
        init_b: d length tensor of biases.
        mat_func: function that returns matrix, log determinant, and inverse
            for linear mapping (see get_LU_map).
        trainable_A: boolean indicating whether to train matrix for linear
            map.
        trainable_b: boolean indicating whether to train bias for linear
            map.
        name: variable scope.
    Returns:
        z: N x d linearly transformed covariates.
        logdet: scalar, the log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer):
        d = int(x.get_shape()[-1])
        if init_mat_params is None:
            # mat_params = tf.get_variable(
            #     'mat_params', dtype=tf.float32,
            #     shape=(d, d), trainable=trainable_A)
            mat_params = tf.get_variable(
                'mat_params', dtype=tf.float32,
                initializer=tf.eye(d, dtype=tf.float32) +
                tf.random_uniform((d, d), -irange, irange),
                trainable=trainable_A)
        else:
            mat_params = tf.get_variable('mat_params', dtype=tf.float32,
                                         initializer=init_mat_params,
                                         trainable=trainable_A)
        if init_b is None:
            # b = tf.get_variable('b', dtype=tf.float32, shape=(d,),
            #                     trainable=trainable_b)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.zeros((d, ), tf.float32),
                                trainable=trainable_b)
        else:
            b = tf.get_variable('b', dtype=tf.float32, initializer=init_b,
                                trainable=trainable_b)
        A, logdet, invmap = mat_func(mat_params, b)
        z = tf.matmul(x, A) + tf.expand_dims(b, 0)
    return z, logdet, invmap


def rnn_coupling(x, rnn_class, name='rnn_coupling'):
    """
    RNN coupling where the covariates are transformed as z_i = x_i + m(s_i).
    Args:
        x: N x d input covariates.
        rnn_class: function the returns rnn_cell with output of spcified size,
            e.g. rnn_class(nout).
        name: variable scope.
    Returns:
        z: N x d rnn transformed covariates.
        logdet: N tensor of log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    with tf.variable_scope(name) as scope:
        # Get RNN cell for transforming single covariates at a time.
        rnn_cell = rnn_class(1)  # TODO: change from 1 to 2 for optional scale
        # Shapes.
        batch_size = tf.shape(x)[0]
        d = int(x.get_shape()[1])
        # Initial variables.
        state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        inp = -tf.ones((batch_size, 1), dtype=tf.float32)
        z_list = []
        for t in range(d):
            m_t, state = rnn_cell(inp, state)
            x_t = tf.expand_dims(x[:, t], -1)
            z_t = x_t + m_t
            z_list.append(z_t)
            inp = x_t
        z = tf.concat(z_list, 1)
        # Jacobian is lower triangular with unit diagonal.
        logdet = 0.0

    # inverse
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            # Shapes.
            batch_size = tf.shape(z)[0]
            # Initial variables.
            state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            inp = -tf.ones((batch_size, 1), dtype=tf.float32)
            x_list = []
            for t in range(d):
                m_t, state = rnn_cell(inp, state)
                z_t = tf.expand_dims(z[:, t], -1)
                x_t = z_t - m_t
                x_list.append(x_t)
                inp = x_t
            x = tf.concat(x_list, 1)
        return x
    return z, logdet, invmap


def leaky_relu(x, alpha):
    return tf.maximum(x, alpha*x)  # Assumes alpha <= 1.0


def general_leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha*tf.nn.relu(-x)


def leaky_transformation(x, alpha=None):
    """Implement an element wise leaky relu transformation."""
    if alpha is None:
        alpha = tf.nn.sigmoid(
            tf.get_variable('log_alpha', initializer=5.0, dtype=tf.float32))
    z = leaky_relu(x, alpha)
    num_negative = tf.reduce_sum(tf.cast(tf.less(z, 0.0), tf.float32), 1)
    logdet = num_negative*tf.log(alpha)

    def invmap(z):
        return tf.minimum(z, z/alpha)

    return z, logdet, invmap


# %% NICE/NVP transformation function.
def additive_coupling(x, hidden_sizes, irange=None, output_irange=None,
                      activation=tf.nn.relu, use_scale=False,
                      scale_function=None, name='additive_coupling'):
    """ NICE additive coupling layer. """
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer) as scope:
        d = int(x.get_shape()[1])
        d_half = d/2
        x_1 = tf.slice(x, [0, 0], [-1, d_half], 'x_1')
        x_2 = tf.slice(x, [0, d_half], [-1, -1], 'x_2')
        if use_scale:
            ms = nn.fc_network(x_2, 2*d_half, hidden_sizes=hidden_sizes,
                               output_init_range=output_irange,
                               activation=activation, name='ms')
            m, s = tf.split(ms, 2, 1)
        else:
            m = nn.fc_network(x_2, d_half, hidden_sizes=hidden_sizes,
                              output_init_range=output_irange,
                              activation=activation, name='m')
            s = tf.zeros_like(m)  # TODO: s = 0.0?
        if scale_function is not None and use_scale:
            fs = scale_function(s)
            y_1 = tf.multiply(x_1, fs) + m
            logdet = tf.reduce_sum(tf.log(tf.abs(fs)), 1)
        else:
            y_1 = tf.multiply(x_1, tf.exp(s)) + m
            logdet = tf.reduce_sum(s, 1)
        y = tf.concat((y_1, x_2), 1, 'y')

    # inverse
    def invmap(y):
        with tf.variable_scope(scope, reuse=True):
            y_1 = tf.slice(y, [0, 0], [-1, d_half], 'y_1')
            y_2 = tf.slice(y, [0, d_half], [-1, -1], 'y_2')
            if use_scale:
                ms = nn.fc_network(y_2, 2*d_half, hidden_sizes=hidden_sizes,
                                   output_init_range=output_irange, reuse=True,
                                   activation=activation, name='ms')
                m, s = tf.split(ms, 2, 1)
            else:
                m = nn.fc_network(y_2, d_half, hidden_sizes=hidden_sizes,
                                  output_init_range=output_irange, reuse=True,
                                  activation=activation, name='m')
                s = tf.zeros_like(m)

            if scale_function is not None and use_scale:
                fs = scale_function(s)
                x_1 = tf.divide(y_1-m, fs)
            else:
                x_1 = tf.divide(y_1-m, tf.exp(s))
            x = tf.concat((x_1, y_2), 1, 'y_inv')
            return x

    return y, logdet, invmap


# %% Conditional based transformation
#
def capped_sigmoid(max_val):
    # Assumes that max_val > 1.0, returns sigmoid where f(0) = 1
    assert max_val > 1.0
    return lambda x: max_val*tf.sigmoid(x - tf.log(max_val-1.0))


def conditioning_transformation(x, conditioning, hidden_sizes,
                                irange=None, output_irange=None,
                                activation=tf.nn.relu,
                                use_scale=True,
                                scale_function=None,
                                name='cond_trans'):
    """
    Transform covariates x using a scaling and shift coming from a fully
    connected network on extranous conditioning information y.
    z = x*exp(s) + m; m,s = split(fc_net(y)).
    Args:
        x: N x d input covariates.
        conditioning: N x p of extraneous conditioning values.
        hidden_sizes: list of hidden layer sizes for use in fc_net for shift
            and scaling.
        irange: scalar, used to initialize the weights of the fc_net randomly
            in [-irange, irange]; a small value helps keep initial
            transformations close to identity.
        output_irange: scalar, seperate initializer to overide irange for the
            output of fc_net.
        activation: activation function to use in fc_net.
        name: variable scope
    Returns:
        z: N x d transformed covariates.
        logdet: scalar, the log determinant of the Jacobian for transformation.
        invmap: function that computes the inverse transformation.
    """
    if conditioning is None:
        # Identity transformation.
        return x, 0.0, (lambda y, c: y)

    # print('\nCCCCCONDITIONING')
    if irange is not None:
        initializer = tf.random_uniform_initializer(-irange, irange)
    else:
        initializer = None
    with tf.variable_scope(name, initializer=initializer) as scope:
        d = int(x.get_shape()[1])
        if use_scale:
            ms = nn.fc_network(conditioning, 2*d, hidden_sizes=hidden_sizes,
                               output_init_range=output_irange,
                               activation=activation, name='ms')
            m, s = tf.split(ms, 2, 1)
        else:
            m = nn.fc_network(conditioning, d, hidden_sizes=hidden_sizes,
                              output_init_range=output_irange,
                              activation=activation, name='ms')
            s = tf.zeros_like(m)  # TODO: s = 0.0?

        if scale_function is not None and use_scale:
            fs = scale_function(s)
            y = tf.multiply(x, fs) + m
            logdet = tf.reduce_sum(tf.log(tf.abs(fs)), 1)
        else:
            y = tf.multiply(x, tf.exp(s)) + m
            logdet = tf.reduce_sum(s, 1)

    # inverse
    def invmap(y, conditioning):
        with tf.variable_scope(scope, reuse=True):
            if use_scale:
                ms = nn.fc_network(
                    conditioning, 2*d, hidden_sizes=hidden_sizes,
                    output_init_range=output_irange,
                    activation=activation, name='ms')
                m, s = tf.split(ms, 2, 1)
            else:
                m = nn.fc_network(conditioning, d, hidden_sizes=hidden_sizes,
                                  output_init_range=output_irange,
                                  activation=activation, name='ms')
                s = tf.zeros_like(m)

            if scale_function is not None and use_scale:
                fs = scale_function(s)
                x = tf.divide(y-m, fs)
            else:
                x = tf.divide(y-m, tf.exp(s))
            return x

    return y, logdet, invmap


###############################################################################
###############################################################################
# %% Set Specific Transformations
def expsum_peq_layer(x, irange=1e-8, name='sum_shift', tau=None):
    """Permutation equivariant transformation with inputs weighted by
       their scaled soft max values:
    z^(i)_j = lambda_j*x^(i)_j + gamma_j*sum_k{exp(tau_j*x^(k)_j)*x^(k)_j}
     / sum_m{exp(tau_j*x^(m)_j)}
    Args:
        x: N x n x d real tensor.
        irange: uniform initialization range of gamma, default 1e-8.
        name: string scope name, default 'sum_shift'.
        tau: the softmax weight (inverse temperature). Can be None,
        a constant or variable vector of shape (1,d). If None, a tf
        parameter of shape (1,d) is initialized using irange.
        Defaults to None.
    Returns:
        z: N x n x d transformed covariates.
        logdet: N real tensor of the log det of the Jacobian.
        invmap: function that computes the inverse transformation.
    """

    def softmax_weighted_sum(x, tau):
        if tau is 0.0:
            # Don't bother creating extraneous nodes
            return tf.reduce_mean(x, axis=1, keepdims=True)
        else:
            y = tf.multiply(x, tf.nn.softmax(tau * x, axis=1))
            # Softmax handles the sum vs mean normalization
            return tf.reduce_sum(y, axis=1, keepdims=True)

    with tf.variable_scope(name) as scope:
        N = tf.shape(x)[1]
        n = tf.shape(x)[1]
        nf = tf.cast(n, tf.float32)
        d = x.get_shape().as_list()[-1]

        lam = tf.get_variable(
            'lambda', dtype=tf.float32, initializer=tf.ones((1, d)))
        initializer = tf.random_uniform_initializer(-irange, irange)
        gam = tf.get_variable(
            'gamma', dtype=tf.float32, shape=(1, d), initializer=initializer)

        if tau is None:
            # We initialize around an equal-weight mean
            tau = tf.get_variable(
                'tau', dtype=tf.float32, shape=(1, d), initializer=initializer)

        # The log determinant is independet of tau and the data
        logdet = tf.reduce_sum(
            tf.cast(n-1, tf.float32) * tf.log(tf.abs(lam)) +
            tf.log(tf.abs(lam + gam)),
            axis=-1)

        logdet_set = logdet * tf.ones(shape=(N))

        weighted_x = softmax_weighted_sum(x, tau)
        z = lam * x + gam * weighted_x

    # Inverse map
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            y = z / lam

            weighted_y = softmax_weighted_sum(y, tau)
            x = y - gam / (lam + gam) * weighted_y
            return x

    return z, logdet, invmap


def linear_peq_layer(x, irange=1e-8, name='linear_peq'):
    """Permutation equivariant linear tranformation of x. It's calculated as
    follows:
        z^(i)_j = lambda_j*x^(i)_j + gamma_j mean_k{x^(k)_j}
    for each set.
    Args:
        x: N x n x d real tensor
        irange: uniform initialization range of gamma, default 1e-8.
        name: string scope name, default 'linear_peq'
    Returns:
        z: N x n x d linearly transformed covariates.
        logdet: N real tensor the log det of the Jacobian.
        invmap: function that computes the inverse transformation.
    """
    return expsum_peq_layer(x, irange=irange, name=name, tau=0.0)


def wsm_peq_layer(x, irange=1e-8, name='weighted_softmax_peq'):
    """Permutation equivariant transformation with inputs weighted by
       their soft max values:
    z^(i)_j = lambda_j*x^(i)_j + gamma_j*sum_k{exp(x^(k)_j)*x^(k)_j}
     / sum_m{exp(x^(m)_j)}
    Args:
        x: N x n x d real tensor.
        irange: uniform initialization range of gamma, default 1e-8.
        name: string scope name, default 'weighted_softmax_peq'.
    Returns:
        z: N x n x d transformed covariates.
        logdet: N real tensor of the log det of the Jacobian.
        invmap: function that computes the inverse transformation.
    """
    return expsum_peq_layer(x, irange=irange, name=name, tau=1.0)


def linear_peq_layer_original(x, irange=1e-8, name='linear_peq'):
    """Permutation equivariant linear tranformation of x. It's calculated a
    follows:
        z^(i)_j = lambda_j*x^(i)_j + gamma_j sum_k*x^(k)_j
    for each set.
    Args:
        x: N x n x d real tensor
        use_mean: Boolean indicating to include 1/n term above.
        name: string scope name.
    Returns:
        z: N x n x d linearly transformed covariates.
        logdet: N real tensor the log det of the Jacobian.
        invmap: function that computes the inverse transformation.
    """

    with tf.variable_scope(name) as scope:
        n = tf.shape(x)[1]
        nf = tf.cast(n, tf.float32)
        d = x.get_shape().as_list()[-1]

        lam = tf.get_variable(
            'lambda', dtype=tf.float32, initializer=tf.ones((1, d)))
        initializer = tf.random_uniform_initializer(-irange, irange)
        gam = tf.get_variable(
            'gamma', dtype=tf.float32, shape=(1, d), initializer=initializer)

        logdet = tf.reduce_sum(
            (nf-1)*tf.log(tf.abs(lam)) + tf.log(tf.abs(lam+nf*gam)),
            axis=-1)

        z = lam*x + gam*tf.reduce_sum(x, axis=1, keepdims=True)

    # Inverse map
    def invmap(z):
        with tf.variable_scope(scope, reuse=True):
            z_sum = tf.reduce_sum(z, axis=1, keepdims=True)
            x = (z/lam) - ((gam/lam)/(lam+nf*gam))*z_sum
            return x

    return z, logdet, invmap


# TODO: check this
def set_nvp(x, hidden_sizes, dim=None, irange=None, output_irange=None,
            activation=tf.nn.relu, use_scale=True, scale_function=None,
            embed_size=256, use_stats=False, conditioning_set_features=None,
            name='set_nvp'):
    """
    TODO
    """
    with tf.variable_scope(name) as scope:
        # Split data.
        d = int(x.get_shape()[2])
        n = tf.shape(x)[1]
        if dim is not None:
            x_c = tf.concat((x[:, :, :dim], x[:, :, dim+1:]), -1)
            x_t = tf.expand_dims(x[:, :, dim], -1)
            t_dim = 1
        else:
            d_half = d/2
            x_c = x[:, :, :d_half]
            x_t = x[:, :, d_half:]
            t_dim = d - d_half
        # Get set features
        if irange is not None:
            initializer = tf.random_uniform_initializer(-irange, irange)
        else:
            initializer = None
        if not use_stats:
            with tf.variable_scope('deepset', initializer=initializer):
                set_feats = us.deepset(x_c, embed_size)
                # set_feats = us.embeding_network(x_c, embed_size=embed_size)
        else:
            set_min = tf.reduce_min(x_c, axis=1)
            set_max = tf.reduce_max(x_c, axis=1)
            set_mean = tf.reduce_mean(x_c, axis=1)
            set_mean_spread = tf.reduce_mean(
                tf.abs(x_c-tf.expand_dims(set_mean, 1)), axis=1)
            set_feats = tf.concat(
                (set_min, set_max, set_mean, set_mean_spread), axis=-1)

        y = us.repeat_set_features(x_c, set_feats)
        if conditioning_set_features is not None:
            y = tf.concat((y, conditioning_set_features), -1)

        # Reshape points, and replicate set feats
        y_dim = int(y.get_shape()[-1])
        # Do conditioning_transformation
        z_t_pnts, logdet_pnts, invmap_pnts = conditioning_transformation(
            tf.reshape(x_t, [-1, t_dim]), tf.reshape(y, [-1, y_dim]),
            hidden_sizes, irange=irange, output_irange=output_irange,
            activation=activation, use_scale=use_scale,
            scale_function=scale_function)
        # Reshape back to sets
        if dim is not None:
            z_t = tf.reshape(z_t_pnts, [-1, n, 1])
            z = tf.concat((x[:, :, :dim], z_t, x[:, :, dim+1:]), -1)
        else:
            z_t = tf.reshape(z_t_pnts, [-1, n, d-d_half])
            z = tf.concat((x_c, z_t), -1)

        # logdet per set
        # TODO: check
        logdet = tf.reduce_sum(tf.reshape(logdet_pnts, [-1, n]), 1)

        def invmap(z, conditioning):
            with tf.variable_scope(scope, reuse=True):
                # Split data.
                d = int(z.get_shape()[2])
                n = tf.shape(z)[1]
                if dim is not None:
                    x_c = tf.concat((z[:, :, :dim], z[:, :, dim+1:]), -1)
                    z_t = tf.expand_dims(z[:, :, dim], -1)
                    t_dim = 1
                else:
                    d_half = d/2
                    x_c = z[:, :, :d_half]
                    z_t = z[:, :, d_half:]
                    t_dim = d - d_half
                # Get set features
                if not use_stats:
                    with tf.variable_scope('deepset', reuse=True):
                        set_feats = us.deepset(x_c, embed_size)
                        # set_feats = us.embeding_network(
                        #     x_c, embed_size=embed_size)
                else:
                    set_min = tf.reduce_min(x_c, axis=1)
                    set_max = tf.reduce_max(x_c, axis=1)
                    set_mean = tf.reduce_mean(x_c, axis=1)
                    set_mean_spread = tf.reduce_mean(
                        tf.abs(x_c-tf.expand_dims(set_mean, 1)), axis=1)
                    set_feats = tf.concat(
                        (set_min, set_max, set_mean, set_mean_spread), axis=-1)

                # Reshape points, and replicate set feats
                y = us.repeat_set_features(x_c, set_feats)
                if conditioning_set_features is not None:
                    y = tf.concat((y, conditioning_set_features), -1)

                y_dim = int(y.get_shape()[-1])
                y_pnts = tf.reshape(y, [-1, y_dim])
                # Do conditioning_transformation
                z_t_pnts = tf.reshape(z_t, [-1, t_dim])
                x_t_pnts = invmap_pnts(z_t_pnts, y_pnts)
                # Reshape back to sets
                if dim is not None:
                    x_t = tf.reshape(x_t_pnts, [-1, n, t_dim])
                    x = tf.concat((z[:, :, :dim], x_t, z[:, :, dim+1:]), -1)
                else:
                    x_t = tf.reshape(x_t_pnts, [-1, n, t_dim])
                    x = tf.concat((x_c, x_t), -1)
            return x

        return z, logdet, invmap


def set_pnt_trans(x, trans_func, conditioning=None, **transargs):
    """
    Applies transformation to each point in the input with optional
    conditioning.
    Args:
        x: N x n x d input covariates.
        trans_func: function that takes a rank 2 tensor as input, and returns
            the transformed inputs, log determinant of the Jacobian and inverse
            map.
        **transargs: Arguments for trans_func.
    Returns:
        z: N x n x d transformed covariates.
        logdet: vector of length N, the log determinant of the Jacobian for the
            transformation of each set.
        invmap: function that computes the inverse transformation.
    """
    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    d = x.get_shape()[2]
    # Reshape to Nn x d.
    x = tf.reshape(x, (N*n, d))
    if conditioning is not None:
        if len(conditioning.get_shape().as_list()) == 2:
            conditioning_pnts = tf.tile(
                tf.expand_dims(conditioning, 1), [1, n, 1])
        else:
            conditioning_pnts = conditioning
        conditioning_pnts = tf.reshape(
            conditioning_pnts, (-1, conditioning_pnts.get_shape()[-1]))
        # Apply transformation.
        print('$$$$$$$$\nSet pnt condintioning:\n{}'.format(conditioning_pnts))
        z, logdet, invmap_rank2 = trans_func(x, conditioning_pnts, **transargs)
    else:
        # Apply transformation.
        z, logdet, invmap_rank2 = trans_func(x, **transargs)
    # Reshape to N x n x d.
    z = tf.reshape(z, (N, n, d))

    # Change inverse map to take in rank 3 tensor.
    def invmap(y, conditioning=None):
        with tf.variable_scope('invmap', reuse=True):
            N = tf.shape(y)[0]
            n = tf.shape(y)[1]
            d = y.get_shape()[2]

            y = tf.reshape(y, (N*n, d))
            if conditioning is not None:
                if len(conditioning.get_shape().as_list()) == 2:
                    conditioning_pnts = tf.tile(
                        tf.expand_dims(conditioning, 1), [1, n, 1])
                else:
                    conditioning_pnts = conditioning
                conditioning_pnts = tf.reshape(
                    conditioning_pnts,
                    (-1, conditioning_pnts.get_shape()[-1]))
                x = invmap_rank2(y, conditioning_pnts)
            else:
                x = invmap_rank2(y)
            x = tf.reshape(x, (N, n, d))
            return(x)
    # Get log determinant of the set, dealing with scalar logdet and tensor
    # logdet separately.
    if len(logdet.shape) is 0 or np.prod(logdet.shape) == 1:
        logdet = tf.reshape(logdet, [1])
        logdet_set = tf.tile(logdet*tf.to_float(n), [N])
    else:
        logdet_set = tf.reshape(tf.expand_dims(logdet, 1), (N, n))
        logdet_set = tf.reduce_sum(logdet_set, axis=1)

    return z, logdet_set, invmap


def corresponding_rnvp(x, corr_cond=None, transform_odd=True,
                       hidden_size=[], irange=None, output_irange=None,
                       activation=tf.nn.relu,
                       use_scale=True, scale_function=None,
                       name='pnt_cnd_nvp'):
    """
    TODO: docstring
    Currently assumes the data has even number of points.
    Transform points in sequences x_i = [x_{i,j}]_{j=1}^n along with side
    conditioning information c_i = [c_{i,j}]_{j=1}^n
        for odd i:
            x_i = s(x_{i-1}, c_{i-1}, c_i) * x_i + m(x_{i-1}, c_{i-1}, c_i)
        for even i:
            x_i = x_i
        (reversed when transform_odd is false)
        or if transform_odd is None:
            x_i = s(c_i) * x_i + m(c_i)
    Args:
        x: N x n x d sequences of points (N sequences of n points of dim d).
        corr_cond: N x n x p sequence of corresponding side information.
        transform_odd: flag indicating whether to transform odd (default) or
            even points. (If None, then transform all points according to
            corr_cond.)
    """
    # TODO: Please check.
    # TODO: Handle odd n
    # TODO: Handle rank 2 case?
    with tf.variable_scope(name) as scope:
        if transform_odd is not None:
            # split into even and odd points [Nxn/2xd]
            x_c, x_t = _split_eo(x) if transform_odd else _split_eo(x)[::-1]
            if corr_cond is not None:
                # TODO: Consider reshape/transpose/reshape for speed
                c_c, c_t = _split_eo(corr_cond) if transform_odd else \
                    _split_eo(corr_cond)[::-1]
                c = tf.concat((x_c, c_c, c_t), axis=-1)
                c_flat = tf.reshape(c, shape=(-1, c.get_shape()[-1]))
            else:
                c_flat = tf.reshape(x_c, shape=(-1, x_c.get_shape()[-1]))
        else:
            if corr_cond is None:
                # Nothing to do, return identity
                return x, 0.0, (lambda y, c: y)
            c_flat = corr_cond
            c_flat = tf.reshape(c_flat, shape=(-1, c_flat.get_shape()[-1]))
            x_t = x

        # Dynamically get the shape so we can set things back after xform
        out_shape = tf.shape(x_t)  # TODO: Do we lose shape info?
        x_t = tf.reshape(x_t, shape=(-1, x_t.get_shape()[-1]), name='x_t_resh')

        z_t, logdet_cond, invmap_pnt = conditioning_transformation(
            x_t, c_flat, hidden_size, irange=irange,
            output_irange=output_irange, activation=activation,
            use_scale=use_scale, scale_function=scale_function)

        z_t = tf.reshape(z_t, shape=out_shape, name='z_t_reshp')
        if transform_odd is not None:
            z = _unsplit_eo(x_c, z_t) if transform_odd else \
                _unsplit_eo(z_t, x_c)
        else:
            z = z_t

        logdet = tf.reduce_sum(tf.reshape(logdet_cond, [-1, out_shape[1]]), 1)

        def invmap(z, corr_cond=None):
            with tf.variable_scope(scope, reuse=True):
                if transform_odd is not None:
                    x_c, z_t = _split_eo(z) if transform_odd else \
                        _split_eo(z)[::-1]
                    if corr_cond is not None:
                        c_c, c_t = _split_eo(corr_cond) if transform_odd else \
                            _split_eo(corr_cond)[::-1]
                        c = tf.concat((x_c, c_c, c_t), axis=-1)
                        c_flat = tf.reshape(c, shape=(-1, c.get_shape()[-1]))
                    else:
                        c_flat = tf.reshape(
                            x_c, shape=(-1, x_c.get_shape()[-1]))
                else:
                    c_flat = corr_cond
                    c_flat = tf.reshape(
                        c_flat, shape=(-1, c_flat.get_shape()[-1]))
                    z_t = z

                # Flatten and invert
                out_shape = tf.shape(z_t)
                z_t = tf.reshape(z_t, shape=(-1, z_t.get_shape()[-1]),
                                 name='z_t_inv_reshp')
                x_t = invmap_pnt(z_t, c_flat)

                # Reshape and reorder
                x_t = tf.reshape(x_t, shape=out_shape, name='x_t_inv_reshp')
                if transform_odd is not None:
                    x = _unsplit_eo(x_c, x_t) if \
                        transform_odd else _unsplit_eo(x_t, x_c)
                else:
                    x = x_t
                return x

        return z, logdet, invmap


def _split_eo(x):
    # TODO:
    # Handle odd second dim
    return x[:, ::2, ...], x[:, 1::2, ...]  # n_odd_flag


def _unsplit_eo(z_c, z_t, shape=None):  # , n_odd_flag):
    # TODO: make work with dynamic n
    # TODO: Remove extra point
    if shape is None:
        d = z_t.get_shape()[-1]
        # n_c = tf.shape(z_c)[1]
        # n_t = tf.shape(z_t)[1]
        n = z_t.get_shape()[1]
        shape = (-1, 2*n, d)
    return tf.reshape(tf.concat((z_c, z_t), axis=-1), shape=shape, name='unsp')
