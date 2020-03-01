import tensorflow as tf  # noqa
import numpy as np  # noqa
from ..utils import nn  # noqa
from . import conditionals as conds  # noqa
from ..model import model as mod


class CondIIDModel(mod.Model):

    # TODO: docstring.
    def __init__(self, base_model,
                 preproc_func=None, name='iid_model'):
        """
        Args:
            transformations: list of transformation functions that take input
                (and possibly conditioning) variables to transform and return
                output, logdet of Jacobian, and inverse for transformation.
            base_model: autoregressive conditional model function that
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
        self.base_model = base_model
        self.preproc_func = preproc_func
        self.name = name

    def build_graph(self, inputs, conditioning=None,
                    sampler_conditioning=None):
        print('IID Model\n\tconditioning {}'.format(
            conditioning))
        # Place holder for model input.
        if self.preproc_func is not None:
            inputs, inv_preproc = self.preproc_func(inputs)
        else:
            inv_preproc = None
        # TODO: remove once can handle dynamic sizes
        # n = tf.shape(inputs)[1]
        self.n = int(inputs.get_shape()[1])
        self.d = int(inputs.get_shape()[2])

        # Sampling extreneous coditioning values.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            # Allows for sampling procedure to be independent from any
            # placeholder/input.
            assert conditioning is not None  # Need to also train conditioning.

        with tf.variable_scope(self.name):
            # Reshape into one rank 2 batch
            inputs_pnts = tf.reshape(inputs, (-1, self.d))
            if conditioning is not None:
                conditioning_pnts = tf.reshape(
                    conditioning, (-1, int(conditioning.get_shape()[-1])))
                sampler_conditioning_pnts = tf.reshape(
                    sampler_conditioning,
                    (-1, int(sampler_conditioning.get_shape()[-1])))
            else:
                conditioning_pnts = None
                sampler_conditioning_pnts = None
            # Get likelihood of each point
            self.llikes, self.sampler = \
                self.base_model.build_graph(
                    inputs_pnts, conditioning_pnts, sampler_conditioning_pnts)
            # Reshape back to rank 3
            self.llikes = tf.reduce_sum(
                tf.reshape(self.llikes, (-1, self.n)), -1)
            self.nll = -tf.reduce_mean(self.llikes)
            self.sampler = tf.reshape(
                self.sampler, (-1, tf.shape(sampler_conditioning)[1], self.d))

        if inv_preproc is not None:
            self.sampler = inv_preproc(self.sampler)

        return self.llikes, self.sampler
