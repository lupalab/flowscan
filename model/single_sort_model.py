import tensorflow as tf  # noqa
import numpy as np  # noqa
from ..utils import nn  # noqa
from . import transforms as trans
from . import conditionals as conds  # noqa
from ..model import model as mod


class SingleSortModel(mod.Model):

    # TODO: docstring.
    def __init__(self, transformations, seq_model,
                 dropout_keep_prob=None,
                 preproc_func=None, name='single_sort',
                 mean_nll=True, trans_conditioning=True,
                 conditional_conditioning=True,
                 ):
        """
        Args:
            transformations: list of transformation functions that take input
                (and possibly conditioning) variables to transform and return
                output, logdet of Jacobian, and inverse for transformation.
            seq_model: autoregressive conditional model function that
                takes in padded inputs (-1, and the first d-1 covariates),
                function to feed hidden states into (see param_func below),
                conditioning values.
            dropout_keep_prob:
            nparams:
            preproc_func:
            name:
            base_distribution:
            trans_conditioning:
            conditional_conditioning:
        """
        # Parameters
        self.transformations = transformations
        self.seq_model = seq_model
        # TODO: use
        self.dropout_keep_prob = dropout_keep_prob
        self.preproc_func = preproc_func
        self.name = name
        self.trans_conditioning = trans_conditioning
        # TODO: use
        self.mean_nll = mean_nll

    def build_graph(self, inputs, conditioning=None,
                    sampler_conditioning=None):
        print('Building {} Graph,\n\tconditioning {}'.format(
            self.name, conditioning))
        # Place holder for model input.
        if self.preproc_func is not None:
            inputs, inv_preproc = self.preproc_func(inputs)
        else:
            inv_preproc = None
        # TODO: remove once can handle dynamic sizes
        self.n = int(inputs.get_shape()[1])
        self.d = int(inputs.get_shape()[2])
        # N = tf.shape(inputs)[0]
        # n = tf.shape(inputs)[1]
        log_n_fac = tf.reduce_sum(
            tf.log(tf.cast(tf.range(self.n), tf.float32)+1.0))

        # Sampling extreneous coditioning values.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            # Allows for sampling procedure to be independent from any
            # placeholder/input.
            assert conditioning is not None  # Need to also train conditioning.

        # Do transformation on input variables.
        if self.transformations is not None:
            with tf.variable_scope('set_transformations') as trans_scope:
                self.z, self.logdet, self.invmap = trans.transformer(
                    inputs, self.transformations,
                    conditioning if self.trans_conditioning else None)
        else:
            self.z = inputs
            self.logdet = 0.0

        # Get conditional parameters, feed through more layers
        self.llikes = self.logdet
        # TODO: make initializer option
        # irange = 1e-1
        # initializer = tf.random_uniform_initializer(-irange, irange)
        # print('using {} sort init'.format(irange))
        initializer = None
        # conditionals of each dimension for set
        # TODO: sort by other function?
        # TODO: shuffle for inverse function?
        with tf.variable_scope('sorted_model', initializer=initializer):
            # Sort the covariates
            # TODO: avoid reshape with dynamic n
            sorted_z = tf.reshape(sort_sets_by_dim(self.z),
                                  (-1, self.n, self.d))

            self.slikes, self.z_samples = \
                self.seq_model.build_graph(
                    sorted_z, conditioning, sampler_conditioning)
            self.llikes += self.slikes - log_n_fac

        # TODO: check.
        # Invert to get samples back in original space.
        if self.transformations is not None:
            with tf.variable_scope(trans_scope, reuse=True):
                self.sampler = self.invmap(
                    self.z_samples,
                    sampler_conditioning if self.trans_conditioning else None)
        else:
            self.sampler = self.z_samples
        if inv_preproc is not None:
            self.sampler = inv_preproc(self.sampler)
        # self.sampler = self.z_samples  # TODO: Remove

        return self.llikes, self.sampler


def sort_sets_by_dim(x, dim=0):
    # TODO: Sort by function value
    x_as = tf.contrib.framework.argsort(x, axis=1)[:, :, dim]
    x_sorted = tf.batch_gather(x, x_as)
    return x_sorted
