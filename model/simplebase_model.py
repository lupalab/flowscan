import tensorflow as tf  # noqa
import numpy as np  # noqa
from ..utils import nn  # noqa
import transforms as trans
import conditionals as conds  # noqa
import likelihoods as likes
from ..model import model as mod


class SimpleBaseModel(mod.Model):

    # TODO: docstring.
    def __init__(self, transformations,
                 preproc_func=None, base_distribution='gaussian',
                 sample_size=128, sample_size_n=1000,
                 trans_conditioning=True,
                 ):
        """
        Args:
            transformations: list of transformation functions that take input
                (and possibly conditioning) variables to transform and return
                output, logdet of Jacobian, and inverse for transformation.
            preproc_func:
            base_distribution:
            sample_size:
            trans_conditioning:
        """
        # Parameters
        self.transformations = transformations
        self.base_distribution = base_distribution
        self.sample_size = sample_size
        self.sample_size_n = sample_size_n
        self.preproc_func = preproc_func
        self.trans_conditioning = trans_conditioning

    def build_graph(self, inputs, conditioning=None,
                    sampler_conditioning=None, forward_tensors=None):
        print('Building {} Graph,\n\tconditioning {}'.format(
            'SimpleBase', conditioning))
        # Place holder for model input.
        if self.preproc_func is not None:
            inputs, inv_preproc = self.preproc_func(inputs)
        else:
            inv_preproc = None
        N = tf.shape(inputs)[0]
        self.d = int(inputs.get_shape()[2])

        # Sampling extreneous coditioning values.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            # Allows for sampling procedure to be independent from any
            # placeholder/input.
            assert conditioning is not None  # Need to also train conditioning.

        # Do transformation on input variables.
        with tf.variable_scope('transformations') as trans_scope:
            self.z, self.logdet, self.invmap = trans.transformer(
                inputs, self.transformations,
                conditioning if self.trans_conditioning else None,
                forward_tensors=forward_tensors
            )

        # Get the likelihood of covariates all iid according to base distro
        # Note: the 3 below is for the weight, mu, sigma param of mixture
        # component and not dimensionality.
        self.llikes = self.logdet
        with tf.variable_scope('conditionals'):
            # Treat as N x nd flat covariates
            flat_z = tf.reshape(self.z, shape=(N, -1, 1))
            std_params = tf.tile(tf.zeros_like(flat_z), [1, 1, 3])
            # Get likelihood with base distribution
            self.llikes += tf.reduce_sum(likes.mixture_likelihoods(
                std_params, flat_z, self.base_distribution), -1)
            # Sample all tensor dimensions iid from base distribution
            total_dims = self.sample_size*self.sample_size_n*self.d
            self.z_samples = tf.reshape(
                conds.sample_mm(
                    tf.zeros(shape=(total_dims, 3), dtype=tf.float32),
                    self.base_distribution),
                (self.sample_size, self.sample_size_n, self.d))

        # Invert to get samples back in original space.
        with tf.variable_scope(trans_scope, reuse=True):
            self.sampler = self.invmap(
                self.z_samples,
                sampler_conditioning if self.trans_conditioning else None)
        if inv_preproc is not None:
            self.sampler = inv_preproc(self.sampler)

        return self.llikes, self.sampler
