import tensorflow as tf  # noqa
import numpy as np  # noqa
from ..utils import nn  # noqa
from . import transforms as trans
from . import conditionals as conds  # noqa
from ..model import model as mod


class CorrespondingModel(mod.Model):

    # TODO: docstring.
    def __init__(self, seq_model,
                 preproc_func=None, name='corresponding_model',
                 irange=None, num_corr_trans=1, hidden_sizes=[],
                 use_scale=True, scale_function=None,
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
        self.seq_model = seq_model
        self.preproc_func = preproc_func
        self.irange = irange
        self.num_corr_trans = num_corr_trans
        self.hidden_sizes = hidden_sizes
        self.use_scale = use_scale
        self.scale_function = scale_function
        self.name = name

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

        # Get conditional parameters, feed through more layers
        with tf.variable_scope(self.name):
            print('\t{} EVEN/ODD transformations (hidden: {}, use_scale:{}, '
                  'sf: {})'
                  '\n\tconditioning:\n\t{}\n\tsampling_cond:\n\t{}\n'.format(
                      self.num_corr_trans, self.hidden_sizes,
                      self.use_scale, self.scale_function,
                      conditioning, sampler_conditioning))

            def corr_trans_cond(x, c):
                return trans.corresponding_rnvp(
                    x, c, transform_odd=None,
                    hidden_size=self.hidden_sizes, irange=self.irange,
                    use_scale=self.use_scale,
                    scale_function=self.scale_function,
                )

            def corr_trans_odd(x, c):
                return trans.corresponding_rnvp(
                    x, c, transform_odd=True,
                    hidden_size=self.hidden_sizes, irange=self.irange,
                    use_scale=self.use_scale,
                    scale_function=self.scale_function,
                )

            def corr_trans_even(x, c):
                return trans.corresponding_rnvp(
                    x, c, transform_odd=False,
                    hidden_size=self.hidden_sizes, irange=self.irange,
                    use_scale=self.use_scale,
                    scale_function=self.scale_function,
                )

            with tf.variable_scope('transformations') as corr_trans_scope:
                self.corr_z, self.corr_logdet, self.corr_invmap = \
                    trans.transformer(
                        inputs,
                        ([] if conditioning is None else [corr_trans_cond]) +
                        [corr_trans_odd, corr_trans_even]*self.num_corr_trans,
                        conditioning)
                self.llikes = self.corr_logdet

            self.slikes, self.corr_z_samples = \
                self.seq_model.build_graph(
                    self.corr_z, conditioning, sampler_conditioning)
            self.llikes += self.slikes

            with tf.variable_scope(corr_trans_scope, reuse=True):
                self.sampler = self.corr_invmap(
                    self.corr_z_samples, sampler_conditioning)

        if inv_preproc is not None:
            self.sampler = inv_preproc(self.sampler)

        return self.llikes, self.sampler
