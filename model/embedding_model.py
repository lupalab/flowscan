import tensorflow as tf  # noqa
import numpy as np  # noqa
from ..utils import nn  # noqa
from . import conditionals as conds  # noqa
from ..model import model as mod
from ..utils import set as uset


class EmbeddingModel(mod.Model):

    # TODO: docstring.
    def __init__(self, base_model, name='embeding_model',
                 irange=None,  embed_size=128, embed_layers=256,
                 ):
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
        self.irange = irange
        self.embed_size = embed_size
        self.embed_layers = embed_layers
        self.name = name

    # Assumes conditioning is rank 3
    def build_graph(self, inputs, conditioning=None,
                    sampler_conditioning=None):
        print('\t\tEMBEDDING_MODEL\n'
              '\t\tconditioning:\n\t\t{}\n\t\tsampling_cond:\n\t\t{}\n'.format(
                  conditioning, sampler_conditioning))

        # TODO: remove once can handle dynamic sizes
        # n = tf.shape(inputs)[1]
        self.n = int(inputs.get_shape()[1])

        # Sampling extreneous coditioning values.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            # Allows for sampling procedure to be independent from any
            # placeholder/input.
            assert conditioning is not None  # Need to also train conditioning.

        # Get conditional parameters, feed through more layers
        with tf.variable_scope(self.name):
            if conditioning is not None:
                if self.irange is not None:
                    initializer = tf.random_uniform_initializer(
                        -self.irange, self.irange)
                else:
                    initializer = None

                with tf.variable_scope(
                        'embed', initializer=initializer) as embed_scope:
                    emb_cond = uset.embeding_network(
                        conditioning, self.embed_layers, self.embed_size)
                    emb_cond = tf.tile(
                        tf.expand_dims(emb_cond, 1), [1, self.n, 1])
                    conditioning = tf.concat((conditioning, emb_cond), -1)
                with tf.variable_scope(embed_scope, reuse=True):
                    emb_scond = uset.embeding_network(
                        sampler_conditioning, self.embed_layers,
                        self.embed_size)
                    emb_scond = tf.tile(
                        tf.expand_dims(emb_scond, 1), [1, self.n, 1])
                    sampler_conditioning = tf.concat(
                        (sampler_conditioning, emb_scond), -1)

            self.llikes, self.sampler = self.base_model.build_graph(
                inputs, conditioning, sampler_conditioning)

        return self.llikes, self.sampler
