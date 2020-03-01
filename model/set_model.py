import tensorflow as tf  # noqa
import numpy as np  # noqa
from ..utils import nn  # noqa
import transforms as trans
import conditionals as conds  # noqa
import set_conditionals as setconds
from ..utils import set as us
from ..model import model as mod


class SetTANModel(mod.Model):

    # TODO: docstring.
    def __init__(self, transformations, conditional_model,
                 param_nlayers=None,
                 hidden_activation=tf.nn.relu, cond_param_irange=None,
                 dropout_keep_prob=None, nparams=None,
                 preproc_func=None, name='tan', base_distribution='gaussian',
                 sample_size=128, sample_size_n=1000,
                 K_0=100, K=5, use_ind_comp=False,
                 conditional_hidden=[256, 256], conditional_embed=128,
                 mean_nll=True, trans_conditioning=True,
                 mean_conditioning=False, conditional_conditioning=True,
                 fc_conditioning=True, no_conditioning_set=False
                 ):
        """
        Args:
            transformations: list of transformation functions that take input
                (and possibly conditioning) variables to transform and return
                output, logdet of Jacobian, and inverse for transformation.
            conditional_model: autoregressive conditional model function that
                takes in padded inputs (-1, and the first d-1 covariates),
                function to feed hidden states into (see param_func below),
                conditioning values.
            param_nlayers: int, number of layers to feed conditional_model's
                hidden state through.
            hidden_activation: activation function to apply to conditional
                hidden states before applying param_func.
            cond_param_irange: scalar range of uniform random initializer for
                hidden state param_func.
            dropout_keep_prob:
            nparams:
            preproc_func:
            name:
            base_distribution:
            sample_size:
            trans_conditioning:
            conditional_conditioning:
            fc_conditioning:
            no_conditioning_set:
        """
        # Parameters
        self.transformations = transformations
        self.conditional_model = conditional_model
        self.param_nlayers = param_nlayers
        self.hidden_activation = hidden_activation
        self.cond_param_irange = cond_param_irange
        self.dropout_keep_prob = dropout_keep_prob
        self.nparams = nparams
        self.base_distribution = base_distribution
        self.sample_size = sample_size
        self.preproc_func = preproc_func
        self.name = name
        self.trans_conditioning = trans_conditioning
        self.sample_size_n = sample_size_n
        self.K_0 = K_0
        self.K = K
        self.use_ind_comp = use_ind_comp
        self.conditional_embed = conditional_embed
        self.condition_hidden = conditional_hidden
        self.mean_nll = mean_nll
        self.mean_conditioning = mean_conditioning
        self.no_conditioning_set = no_conditioning_set

    def build_graph(self, inputs, conditioning=None,
                    sampler_conditioning=None):
        print('Building {} Graph,\n\tconditioning {}'.format(
            self.name, conditioning))
        # Place holder for model input.
        if self.preproc_func is not None:
            inputs, inv_preproc = self.preproc_func(inputs)
        else:
            inv_preproc = None
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        self.d = int(inputs.get_shape()[2])

        # Sampling extreneous coditioning values.
        if sampler_conditioning is None and not self.no_conditioning_set:
            sampler_conditioning = conditioning
        elif self.no_conditioning_set:
            sampler_conditioning = None
            conditioning = None
        else:
            # Allows for sampling procedure to be independent from any
            # placeholder/input.
            assert conditioning is not None  # Need to also train conditioning.

        # Do transformation on input variables.
        with tf.variable_scope('transformations') as trans_scope:
            self.z, self.logdet, self.invmap = trans.transformer(
                inputs, self.transformations,
                conditioning if self.trans_conditioning else None)

        # Get conditional parameters, feed through more layers
        self.llikes = self.logdet
        self.z_samples = None
        result_ll_list = [None for _ in range(self.d)]
        # TODO: make initializer option
        irange = 1e-6
        initializer = tf.random_uniform_initializer(-irange, irange)
        # conditionals of each dimension for set
        with tf.variable_scope('conditionals', initializer=initializer):
            for di in range(self.d):
                with tf.variable_scope('dimension_{}'.format(di)):
                    cond_input = tf.expand_dims(self.z[:, :, di], -1)
                    if di > 0 and not self.no_conditioning_set:
                        cond_set = self.z[:, :, :di]
                        K = self.K
                    elif di > 0:
                        cond_set = None
                        K = self.K
                    else:
                        cond_set = None
                        K = self.K_0
                    print('Dimension {}, K {}'.format(di, K))
                    if self.no_conditioning_set:
                        set_embed_func = None
                    else:
                        set_embed_func = lambda X: us.deepset(X, 128)
                    result_ll, sampler_di = setconds.set_conditional(
                        cond_input,
                        set_embed_func=lambda X: us.deepset(
                            X, self.conditional_embed),
                        conditioning_set=cond_set,
                        conditioning_set_features=conditioning,
                        base_distribution=self.base_distribution,
                        K=K, nparams=self.nparams,
                        hidden_sizes=self.condition_hidden,
                        use_ind_comp=self.use_ind_comp,
                        mean_conditioning=self.mean_conditioning,
                    )
                    if self.no_conditioning_set:
                        z_samp_di = sampler_di(
                            self.sample_size, sample_n=self.sample_size_n,
                            sampler_conditioning_set=None,
                            sampler_conditioning_set_features=None)
                    else:
                        z_samp_di = sampler_di(
                            self.sample_size, sample_n=self.sample_size_n,
                            sampler_conditioning_set=self.z_samples,
                            sampler_conditioning_set_features=sampler_conditioning)
                    if di > 0:
                        self.z_samples = tf.concat(
                            (self.z_samples, z_samp_di), -1)
                    else:
                        self.z_samples = z_samp_di
                    result_ll_list[di] = result_ll

        print(result_ll_list)
        self.llikes += sum(result_ll_list)
        if self.mean_nll:
            self.nll = -tf.reduce_sum(self.llikes)/tf.cast(N*n, tf.float32)
        else:
            self.nll = -tf.reduce_sum(self.llikes)

        # TODO: check.
        # Invert to get samples back in original space.
        with tf.variable_scope(trans_scope, reuse=True):
            self.sampler = self.invmap(
                self.z_samples,
                sampler_conditioning if self.trans_conditioning else None)
        if inv_preproc is not None:
            self.sampler = inv_preproc(self.sampler)

        return self.nll, self.llikes, self.sampler
