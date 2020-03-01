import itertools
import tensorflow as tf
import numpy as np


def gaussian_mean_loss(sample, do_l1=False):
    with tf.variable_scope('gaussian_mean'):
        means = tf.reduce_mean(sample, axis=0)
        if do_l1:
            penalties = tf.abs(means)
        else:
            penalties = tf.square(means)
        penalty = tf.reduce_mean(penalties)
    return penalty


def mean_loss(sample, sample2=None, do_l1=False):
    with tf.variable_scope('gaussian_mean'):
        means = tf.reduce_mean(sample, axis=0)
        if sample2 is None:
            means2 = tf.zeros_like(means)
        else:
            means2 = tf.reduce_mean(sample2, axis=0)
        if do_l1:
            penalties = tf.abs(means-means2)
        else:
            penalties = tf.square(means-means2)
        penalty = tf.reduce_mean(penalties)
    return penalty


def gaussian_cov_loss(sample, central=False, do_l1=False, only_diag=False):
    with tf.variable_scope('gaussian_cov'):
        in_shape = tf.shape(sample)
        N, d = tf.cast(in_shape[0], tf.float32), in_shape[1]
        if central:
            means = tf.reduce_mean(sample, axis=0)
            centered = sample - means
            cov = tf.matmul(centered, centered, transpose_a=True)/N
        else:
            cov = tf.matmul(sample, sample, transpose_a=True)/N
        eye = tf.eye(d)
        d = tf.cast(d, tf.float32)
        diff = cov - eye
        if do_l1:
            penalties = tf.abs(diff)
        else:
            penalties = tf.square(diff)
        if only_diag:
            penalties = tf.matrix_band_part(penalties, 0, -1)
            nelems = d
        else:
            penalties = tf.matrix_band_part(penalties, 0, 0) + \
                tf.matrix_band_part(penalties, 0, -1)
            nelems = d + d*(d-1)
        penalty = tf.reduce_sum(penalties)/nelems
    return penalty


def cov_loss(sample, sample2=False, central=False, do_l1=False,
             only_diag=False, only_offdiag=False, name='cov'):
    with tf.variable_scope(name):
        in_shape = tf.shape(sample)
        N, d = tf.cast(in_shape[0], tf.float32), in_shape[1]
        if central:
            means = tf.reduce_mean(sample, axis=0)
            centered = sample - means
            cov = tf.matmul(centered, centered, transpose_a=True)/N
            if sample2 is None:
                cov2 = tf.eye(d)
            else:
                N2 = tf.cast(tf.shape(sample2)[0], tf.float32)
                means2 = tf.reduce_mean(sample2, axis=0)
                centered2 = sample2 - means2
                cov2 = tf.matmul(centered2, centered2, transpose_a=True)/N2
        else:
            cov = tf.matmul(sample, sample, transpose_a=True)/N
            if sample2 is None:
                cov2 = tf.eye(d)
            else:
                N2 = tf.cast(tf.shape(sample2)[0], tf.float32)
                means2 = tf.reduce_mean(sample2, axis=0)
                cov2 = tf.matmul(means2, means2, transpose_a=True)/N2
        d = tf.cast(d, tf.float32)
        diff = cov - cov2
        if do_l1:
            penalties = tf.abs(diff)
        else:
            penalties = tf.square(diff)
        if only_diag:
            penalties = tf.matrix_band_part(penalties, 0, -1)
            nelems = d
        elif only_offdiag:
            penalties = tf.matrix_band_part(penalties, 0, -1)
            nelems = d*(d-1)
        else:
            penalties = tf.matrix_band_part(penalties, 0, 0) + \
                tf.matrix_band_part(penalties, 0, -1)
            nelems = d + d*(d-1)
        penalty = tf.reduce_sum(penalties)/nelems
    return penalty


def rdir_cov_loss(sample, D=100, sigma=1.0, frequencies=None):
    d = int(sample.get_shape()[1])
    with tf.variable_scope('rdir_cov'):
        if frequencies is None:
            frequencies = tf.random_normal((d, D), stddev=sigma)
        rand_product = tf.matmul(sample, frequencies)
        rand_feats = tf.concat((tf.cos(rand_product), tf.sin(rand_product)),
                               1)/tf.sqrt(D)
        u = tf.random_normal((2*D, 1))
        u /= tf.norm(u)
        rand_func = tf.matmul(rand_feats, u)
        return cov_loss(rand_func, only_offdiag=True)


def mmd_loss(sample, sample2=None, D=100, sigma=1.0, frequencies=None,
             random_projection=False, use_basis=False):
    """Calculate an approximate mmd of sample distribution to a Gaussian.
    Args:
        sample: Tensor (? x d), sample from non-Gaussian distribution.
        D: int, number for random frequencies to sample
        sigma: float, the standard deviation of random frequencies.
    """
    with tf.variable_scope('mmd'):
        d = int(sample.get_shape()[1])
        if random_projection:
            if use_basis:
                eye = tf.eye(d)
                u = tf.transpose(tf.gather(
                    eye,
                    tf.random_uniform((1,), maxval=d, dtype=tf.int64)
                ))
            else:
                u = tf.random_normal((d, 1))
                u /= tf.norm(u)
            inps = tf.matmul(sample, u)
            if sample2 is None:
                inps2 = None
            else:
                inps2 = tf.matmul(sample2, u)
            d = 1
        else:
            inps = sample
            if sample2 is None:
                inps2 = None
            else:
                inps2 = sample2
        if frequencies is None:
            frequencies = tf.random_normal((d, D), stddev=sigma)
        rand_product = tf.matmul(inps, frequencies)
        mu_cos = tf.reduce_mean(tf.cos(rand_product), 0)
        mu_sin = tf.reduce_mean(tf.sin(rand_product), 0)
        if inps2 is None:
            mu_cos2 = tf.exp(-0.5*tf.reduce_sum(tf.square(frequencies), 0))
            mu_sin2 = tf.zeros_like(mu_cos2)
        else:
            rand_product2 = tf.matmul(inps2, frequencies)
            mu_cos2 = tf.reduce_mean(tf.cos(rand_product2), 0)
            mu_sin2 = tf.reduce_mean(tf.sin(rand_product2), 0)
        mmd = tf.reduce_mean(tf.squared_difference(mu_sin, mu_sin2)) + \
            tf.reduce_mean(tf.squared_difference(mu_cos, mu_cos2))
    return mmd


def pair_hsic(sample, D=16, sigma=1.0, frequencies=None,
              gaussian_marginals=False, name='indep_mmd'):
    """Calculate an approximate mmd of sample distribution to a Gaussian.
    Args:
        sample: Tensor (? x d), sample from non-Gaussian distribution.
        D: int, number for random frequencies to sample
        sigma: float, the standard deviation of random frequencies.
    """
    # TODO: Check this!
    with tf.variable_scope(name):
        # Get 2 random dimensions
        N = tf.cast(tf.shape(sample)[0], tf.float32)
        d = int(sample.get_shape()[1])
        pairs_arr = np.array([c for c in itertools.combinations(range(d), 2)])
        pairs = tf.constant(pairs_arr, dtype=tf.int64, name='pairs')
        rind = tf.random_uniform([], maxval=d*(d-1)/2, dtype=tf.int64)
        rpair = tf.gather(pairs, rind)
        x = tf.transpose(tf.gather(tf.transpose(sample), rpair[0]))
        y = tf.transpose(tf.gather(tf.transpose(sample), rpair[1]))
        # Get random features on each dimension
        if frequencies is None:
            frequencies = tf.random_normal((1, D), stddev=sigma)
        rand_product = tf.matmul(tf.expand_dims(x, -1), frequencies)
        rf_x = tf.concat((tf.cos(rand_product), tf.sin(rand_product)), 1)
        rand_product2 = tf.matmul(tf.expand_dims(y, -1), frequencies)
        rf_y = tf.concat((tf.cos(rand_product2), tf.sin(rand_product2)), 1)
        # Compute differnce on mean outerproduct and outproduct of means
        mme_xy = tf.matmul(rf_x, rf_y, transpose_a=True)/N
        if gaussian_marginals:
            mu_cos = tf.expand_dims(
                tf.exp(-0.5*tf.reduce_sum(tf.square(frequencies), 0)), 0
            )
            mu_sin = tf.zeros_like(mu_cos)
            mme_x = tf.concat((mu_cos, mu_sin), 1)
            mme_y = mme_x
        else:
            mme_x = tf.reduce_mean(rf_x, axis=0, keep_dims=True)
            mme_y = tf.reduce_mean(rf_y, axis=0, keep_dims=True)
        mme_outer = tf.matmul(mme_x, mme_y, transpose_a=True)
        mmd = tf.reduce_mean(tf.squared_difference(mme_xy, mme_outer))
    return mmd


def gaussian_mmd_loss(sample, D=100, sigma=1.0, frequencies=None,
                      random_projection=False, use_basis=False):
    """Calculate an approximate mmd of sample distribution to a Gaussian.
    Args:
        sample: Tensor (? x d), sample from non-Gaussian distribution.
        D: int, number for random frequencies to sample
        sigma: float, the standard deviation of random frequencies.
    """
    with tf.variable_scope('gaussian_mmd'):
        d = int(sample.get_shape()[1])
        if random_projection:
            if use_basis:
                eye = tf.eye(d)
                u = tf.transpose(tf.gather(
                    eye,
                    tf.random_uniform((1,), maxval=d, dtype=tf.int64)
                ))
            else:
                u = tf.random_normal((d, 1))
                u /= tf.norm(u)
            inps = tf.matmul(sample, u)
            d = 1
        else:
            inps = sample
        if frequencies is None:
            frequencies = tf.random_normal((d, D), stddev=sigma)
        rand_product = tf.matmul(inps, frequencies)
        mu_cos = tf.reduce_mean(tf.cos(rand_product), 0)
        mu_sin = tf.reduce_mean(tf.sin(rand_product), 0)
        exp_freqs = tf.exp(-0.5*tf.reduce_sum(tf.square(frequencies), 0))
        mmd = tf.reduce_mean(tf.square(mu_sin)) + \
            tf.reduce_mean(tf.squared_difference(mu_cos, exp_freqs))
    return mmd


def linear_transform(x, do_diagonal=False, epsilon=0.0, name='lintrans',
                     reuse=None, identity_init=False):
    with tf.variable_scope(name, reuse=reuse):
        ncode = int(x.get_shape()[-1])
        if do_diagonal:
            init = None if not identity_init else \
                tf.ones((ncode,), dtype=tf.float32)
            A = tf.get_variable('A_diag', shape=(ncode,), dtype=tf.float32,
                                initializer=init)
            A = tf.diag(tf.square(A))
        else:
            init = None if not identity_init else tf.eye(ncode)
            A = tf.get_variable('A_sqrt', shape=(ncode, ncode),
                                dtype=tf.float32, initializer=init)
            A = tf.matmul(A, A, transpose_b=True)
        if epsilon > 0.0:
            A += epsilon*tf.eye(ncode)
        b = tf.get_variable('b', dtype=tf.float32,
                            initializer=tf.zeros((1, ncode), dtype=tf.float32))
        z = tf.matmul(x, A) + b
    return z


def gaussian_score(codes,
                   do_ztransform=False, do_diagonal=False, epsilon=0.0,
                   D=200, sigma=1.0, use_multid_mme=False, gpen=1.0,
                   n_hsic=16, gaussian_hsic=True, hpen=1.0
                   ):
    # Apply a linear transformation to codes? Equivalent to checking that codes
    # are coming from non-standard Gaussian (with inverse transormation).
    if do_ztransform:
        zcodes = linear_transform(codes, do_diagonal=do_diagonal,
                                  epsilon=epsilon)
    else:
        zcodes = codes

    # Compute moment and mmd based penalties.
    loss = 0.0
    frequencies = None
    loss_mmd = gaussian_mmd_loss(zcodes, D=D, sigma=sigma,
                                 frequencies=frequencies,
                                 random_projection=True)
    loss_mmd_marg = gaussian_mmd_loss(zcodes, D=D, sigma=sigma,
                                      frequencies=frequencies,
                                      random_projection=True,
                                      use_basis=True)
    loss_mom1 = gaussian_mean_loss(zcodes, do_l1=False)
    loss_mom2 = gaussian_cov_loss(zcodes, do_l1=False, central=True)
    gaussian_pens = (loss_mmd_marg, loss_mmd, loss_mom1, loss_mom2)

    if use_multid_mme:
        loss_multi_mmd = gaussian_mmd_loss(
            zcodes, D=D, sigma=sigma, frequencies=frequencies,
            random_projection=False
        )
        gaussian_pens += (loss_multi_mmd,)
    loss += gpen*sum(gaussian_pens)

    if n_hsic > 0:
        hsic_pens = []
        for nimmd in range(n_hsic):
            hsic_pens += [
                pair_hsic(zcodes, D=D, sigma=sigma,
                          name='immd{}'.format(nimmd),
                          gaussian_marginals=gaussian_hsic)
            ]
        loss += hpen*sum(hsic_pens)/float(n_hsic)
    else:
        return loss, gaussian_pens
    return loss, gaussian_pens, hsic_pens


def gaussian_likes(codes, use_marginals=False, mean_l2=True, factor=1.0,
                   name='glikes'):
    with tf.variable_scope(name):
        pens = tf.square(codes)
        if not use_marginals:
            pens = tf.reduce_mean(pens, 1) if mean_l2 else \
                tf.reduce_sum(pens, 1)
        pens = tf.reduce_mean(tf.exp(-0.5 * factor * pens))
    return pens
