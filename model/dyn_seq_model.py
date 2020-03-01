import tensorflow as tf
import numpy as np
from . import model as mod
from tqdm import tqdm
# from ..utils import nn


class SeqTANModel(mod.Model):

    def __init__(self, base_model, cell_class, seq_feats=512,
                 preproc_func=None, markov_feat=False,
                 fc_layers=None, sample_size_n=1000, name='seq_stan'):
        self.base_model = base_model
        self.preproc_func = preproc_func
        self.sample_size = self.base_model.sample_size
        self.sample_size_n = int(sample_size_n)
        self.seq_feats = seq_feats
        self.rnn_cell = cell_class(seq_feats)
        # Explicitly keep condition on last point
        self.markov_feat = markov_feat
        self.name = name
        # TODO incorporate initial input layers into RNN cell_class
        self.fc_layers = fc_layers
        print('Seq markov feat {}'.format(self.markov_feat))

    def build_graph(
        self, inputs, conditioning=None, sampler_conditioning=None
    ):
        """ Builds the graph for a structured sequential TAN.
        Args:
            inputs: N x n x d real tensor of batch of N sequences each with
                n points.
            conditioning: N x s or N x n x s tensor of conditioning values
        """
        # TODO: Extend to inputs that are rank > 3.
        if self.preproc_func is not None:
            inputs, inv_preproc = self.preproc_func(inputs)
        else:
            inv_preproc = None
        print('Making sequential model.')
        print(inputs)
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        # Initial states.
        like_state = self.rnn_cell.zero_state(N, dtype=tf.float32)
        samp_state = self.rnn_cell.zero_state(
            self.sample_size, dtype=tf.float32)
        inp_shape = inputs.get_shape().as_list()[2:]
        # External conditioning values for sampling.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            assert conditioning is not None  # Need to also train conditioning.

        with tf.variable_scope(self.name + '_rnncell') as scope:
            # Dynamic RNN to get RNN outputs
            rnn_layer = tf.keras.layers.RNN(self.rnn_cell, return_sequences=True)
            padded_inputs = tf.concat(
                (-tf.ones([N, 1] + inp_shape), inputs), 1)[:, :-1, ...]
            v_like = rnn_layer(padded_inputs)  # N x n x s
        # Reshape to flat vectors Nn x ...
        flat_padded_inputs = tf.reshape(
            padded_inputs, (-1, np.prod(inp_shape)))
        flat_inputs = tf.reshape(inputs, (-1, np.prod(inp_shape)))
        v_like = tf.reshape(v_like, (-1, v_like.get_shape()[-1]))
        if self.markov_feat:
            v_like = tf.concat((v_like, flat_padded_inputs), 1)
        if conditioning is not None:
            cond_shape = conditioning.get_shape().as_list()
            if cond_shape[-1] > 0:
                if len(cond_shape) == 2:
                    conditioning_flat = tf.tile(
                        tf.expand_dims(conditioning, 1), [1, n, 1])
                else:
                    conditioning_flat = conditioning
                conditioning_flat = tf.reshape(
                    conditioning_flat, (-1, cond_shape[-1]))
                v_like = tf.concat((v_like, conditioning_flat), 1)
            else:
                conditioning = None  # ...
                print("cond_shape has shape of zero.")

        # Gather losses, likelihoods across times.
        # TODO: could reshape the sampler here, if the
        # sampler_conditioning was None
        with tf.variable_scope(self.name) as scope:
            likes, _ = self.base_model.build_graph(flat_inputs, v_like)
            likes = tf.reshape(likes, (N, n))
        # Sum and pack.
        llikes = tf.reduce_sum(likes, -1)

        # Gather samples across times.
        # Dummy input
        inp = -tf.ones([N, np.prod(inp_shape)])
        with tf.variable_scope('rnn', reuse=True) as rnn_scope:
            v_like, like_state = self.rnn_cell(inp, like_state)
        if self.markov_feat:
            v_like = tf.concat((v_like, inp), 1)
        if conditioning is not None:
            v_like = tf.concat((v_like, conditioning_flat), 1)
        # TODO: make sure that RNN scope is the same as with keras
        samplers = [None]*self.sample_size_n
        for t in tqdm(range(self.sample_size_n), desc='Building Sampler'):
            # Input will be last seen sample, or -1s.
            if t == 0:
                samp_inp = -tf.ones([self.sample_size]+inp_shape)
            else:
                samp_inp = samplers[t-1]
            # Flatten for use with rnn_cell.
            samp_inp = tf.reshape(samp_inp, [-1, np.prod(inp_shape)])
            # Get state for sampling with TAN.
            with tf.variable_scope(rnn_scope, reuse=True):
                v_samp, samp_state = self.rnn_cell(samp_inp, samp_state)
            if self.markov_feat:
                v_samp = tf.concat((v_samp, samp_inp), 1)
            if sampler_conditioning is not None:
                if len(cond_shape) == 2:
                    v_samp = tf.concat((v_samp, sampler_conditioning), 1)
                else:
                    v_samp = tf.concat(
                        (v_samp, sampler_conditioning[:, t, ...]), 1)
            with tf.variable_scope(scope, reuse=True):
                # Use the base model on this time-step.
                _, samplers[t] = self.base_model.build_graph(
                    inp, v_like, v_samp)
        sampler = tf.stack(samplers, 1)
        if inv_preproc is not None:
            sampler = inv_preproc(sampler)
        print(sampler)

        return llikes, sampler
