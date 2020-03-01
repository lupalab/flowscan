"""likelihoods.py

Logic to get likelihoods given parameters to conditional mixtures.
- `mixture_likelihoods` function produces log likelihoods on transformed
covariates given parameters
- `make_nll_loss` gives negative log likelihood of data in batch
"""

import tensorflow as tf
import numpy as np


# NOTE: CMB
# I think we need to define N, n, and d explicitely in the documentation
# I *believe*, that
#   N is the number of sets (the batch size)
#   n is the number of points in the set
#   d is the dimensionality of each point
# In the non-set cases, n=1 and we (essentially) squeeze it out

def mixture_likelihoods(params, targets, base_distribution='gaussian',
                        name='mixture_likelihood'):
    """Given log-unnormalized mixture weights, shift, and log scale parameters
    for mixture components, return the likelihoods for targets.
    Args:
        params: N x d x 3*ncomp tensor of parameters of mixture model
            where weight_logits, means, log_sigmas = tf.split(params, 3, 2).
        targets: N x d x 1 tensor of 1d targets to get likelihoods for.
        base_distribution: {'gaussian', 'laplace', 'logistic'} the base
            distribution of mixture components.
    Return:
        likelihoods: N x d  tensor of likelihoods.
    """
    base_distribution = base_distribution.lower()
    with tf.variable_scope(name):
        # Compute likelihoods per target and component
        # Write log likelihood as logsumexp.
        logits, means, lsigmas = tf.split(params, 3, 2)
        sigmas = tf.exp(lsigmas)
        if base_distribution == 'gaussian':
            log_norm_consts = -lsigmas - 0.5*np.log(2.0*np.pi)
            log_kernel = -0.5*tf.square((targets-means)/sigmas)
        elif base_distribution == 'laplace':
            log_norm_consts = -lsigmas - np.log(2.0)
            log_kernel = -tf.abs(targets-means)/sigmas
        elif base_distribution == 'logistic':
            log_norm_consts = -lsigmas
            diff = (targets-means)/sigmas
            log_kernel = -tf.nn.softplus(diff) - tf.nn.softplus(-diff)
        else:
            raise NotImplementedError
        log_exp_terms = log_kernel + log_norm_consts + logits
        log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - \
            tf.reduce_logsumexp(logits, -1)
        # log_likelihoods = tf.Print(
        #     log_likelihoods,
        #     [tf.shape(log_exp_terms), tf.shape(params), tf.shape(targets)],
        #     message='shape log_exp_terms, params, targets'
        # )
    return log_likelihoods


# TODO: comments.
# TODO: make more general than single dimension?
def new_mixture_of_mixtures(params, targets, base_distribution='gaussian',
                            name='mixofmix'):
    """Given log-unnormalized mixture weights, shift, and log scale parameters
    for mixture components for K mixtures, return the likelihoods for targets.
    Args:
        params: {N, 1} x K x 3*ncomp+1 tensor of parameters,
            where params[:, :, :-1] are mixture parameters and
            params[:, :, -1] are logit weights.
        targets: N x n x 1 tensor of 1d targets to get likelihoods for.
        base_distribution: {'gaussian', 'laplace', 'logistic'} the base
            distribution of mixture components.
    Return:
        likelihoods: N tensor of likelihoods.
    """
    N = tf.shape(targets)[0]
    n = tf.shape(targets)[1]
    params_shape = params.get_shape().as_list()
    K = params_shape[1]
    mix_pnts_lls_list = [None for _ in range(K)]
    for k in range(K):
        mparams = tf.expand_dims(params[:, k, :-1], 1)
        # ^ {N,1} x 1 x 3*ncomps
        mlls = mixture_likelihoods(mparams, targets, base_distribution)
        # ^ N x n
        mix_pnts_lls_list[k] = mlls
    mix_pnts_lls = tf.stack(mix_pnts_lls_list, -1)  # N x n x k
    mix_lls = tf.reduce_sum(mix_pnts_lls, 1)  # N x k
    weights = params[:, :, -1]  # {N,1} x K
    if params_shape[0] is not None:
        weights = tf.tile(weights, [N, 1])  # N x K
    log_exp_terms = mix_lls + weights
    log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - \
        tf.reduce_logsumexp(weights, -1)
    return log_likelihoods


def mixture_of_mixtures(weights, params, targets, base_distribution='gaussian',
                        name='mixofmix'):
    """Given log-unnormalized mixture weights, shift, and log scale parameters
    for mixture components, return the likelihoods for targets.
    Args:
        weights: K tensor of logits weighing mixture of mixture.
        params: K x 3*ncomp tensor of parameters of mixture models where
            weight_logits, means, log_sigmas = tf.split(params[k, :], 3, 0).
        targets: N x n x 1 tensor of 1d targets to get likelihoods for.
        base_distribution: {'gaussian', 'laplace', 'logistic'} the base
            distribution of mixture components.
    Return:
        likelihoods: N x 1 tensor of likelihoods.
    """
    K = params.get_shape()[0]
    params_k = tf.split(params, K, 0)
    n = tf.shape(targets)[1]
    target_pnts = tf.reshape(targets, (-1, 1, 1))  # Nn x 1 x 1
    # target_pnts = tf.Print(
    #     target_pnts,
    #     [tf.shape(target_pnts), tf.shape(targets)],
    #     message='wtf target_pnts targets'
    # )
    # TODO: tile approach?
    mix_pnts_lls = tf.stack([
        tf.reshape(
            mixture_likelihoods(  # Nn x 1
                tf.expand_dims(pk, 0), target_pnts, base_distribution),
            (-1, n))  # N x n
        for pk in params_k], -1) # N x n x K
    mix_lls = tf.reduce_sum(mix_pnts_lls, 1)
    log_exp_terms = mix_lls + tf.expand_dims(weights, 0)
    log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1, keepdims=False) - \
        tf.reduce_logsumexp(weights)
    # log_likelihoods = tf.Print(
    #     log_likelihoods,
    #     [tf.shape(log_likelihoods), tf.shape(mix_pnts_lls), tf.shape(target_pnts)],
    #     message='shape log_likelihoods mix_pnts_lls, target_pnts'
    # )
    # import pdb; pdb.set_trace()  # XXX BREAKPOINT
    return log_likelihoods


def make_nll_loss( logits, targets, logdetmap, likefunc=mixture_likelihoods,
                  min_like=None, input_is_set=False ):


    if ( input_is_set ):
        # Here we're treating everything as independent
        lshape = tf.get_shape( logits ).as_list()
        logits = tf.reshape( logits, [lshape[0], -1, lshape[-1]] )

        tshape  = tf.get_shape( targets ).as_list()
        targets = tf.reshape( targets, [tshape[0], -1, tshape[-1]] )

    return make_nll_loss_orig( logits, targets, logdetmap, likefunc, min_like )


def make_nll_loss_orig(logits, targets, logdetmap, likefunc=mixture_likelihoods,
                  min_like=None):
    """Given log-unnormalized mixture weights for equi-spaced truncated
    Gaussians on the unit interval, return the likelihoods for targets.
    Args:
        logits: N x d x 3*ncomp tensor of log unnormalized logits to be
            softmaxed for respective weights on mixture components.
        targets: N x d x 1 tensor of 1d targets to get likelihoods for.
        logdetmap: N tensor (or scalar) of determinant normalizers
        likefunc: function to compute conditional log likelihoods on each
            dimension.
        min_like: scalar Minimum likelihood to truncate, (None used by default).
    Return:
        loss: scalar nll on batch.
        ll: N tensor of log likelihoods.
    """
    with tf.variable_scope('nll_loss'):
        lls = log_likelihoods(logits, targets, logdetmap, likefunc=likefunc,
                              min_like=min_like)
        loss = -tf.reduce_mean(lls)
    return loss, lls


def log_likelihoods(logits, targets, logdetmap, likefunc=mixture_likelihoods,
                    min_like=None):
    """Convinience function that returns the unavaraged tensor of log
    likelihoods.
    """
    with tf.variable_scope('ll'):
        cllikes = likefunc(logits, targets)
        mix_ll = tf.reduce_sum(cllikes, -1)
        lls = logdetmap + mix_ll
        if min_like is not None:
            trunc_lls = tf.maximum(lls, np.log(min_like), 'trunc_likes')
        else:
            trunc_lls = lls
    return trunc_lls

