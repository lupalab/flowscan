import tensorflow as tf
import numpy as np  # noqa
import trainer
from ..model import condiid_model as iidmod
from ..model import model as mod
from ..utils import nn  # noqa
from ..utils import set as uset


# TODO: replace with other deepset?
def encoder(inputs, embed_layers, embed_size, const_lsig=True):
    with tf.variable_scope('encoder'):
        Wmu = tf.get_variable(
            'Wmu', shape=(2*embed_size, embed_size), dtype=tf.float32)
        bmu = tf.get_variable(
            'bmu', shape=(embed_size, ), dtype=tf.float32)
        if const_lsig:
            feat_lsig = tf.get_variable(
                'feat_lsig', shape=(1, embed_size), dtype=tf.float32)
        else:
            Wsig = tf.get_variable(
                'Wsig', shape=(2*embed_size, embed_size), dtype=tf.float32)
            bsig = tf.get_variable(
                'bsig', shape=(embed_size, ), dtype=tf.float32)

        embed_feats = uset.embeding_network(inputs, embed_layers, embed_size)
        # Get latent code parameters
        feat_mu = tf.nn.xw_plus_b(embed_feats, Wmu, bmu, 'linearmu')
        if not const_lsig:
            feat_lsig = tf.nn.xw_plus_b(embed_feats, Wsig, bsig, 'linearsig')

        return feat_mu, feat_lsig


class VAEExperiment:

    # TODO: make all arguements optional, load config from
    # save_location+'config.p' when not given, resolve dimension, and add
    # option to restore
    def __init__(self, config, summary_location, save_location, fetchers,
                 inputs_pl=None, conditioning_pl=None):
        self.config = config
        self.summary_location = summary_location
        self.save_location = save_location
        self.fetchers = fetchers

        # Input placeholders
        self.inputs_pl = inputs_pl
        self.conditioning_pl = conditioning_pl

        # TF control
        self.sess = None
        self.fixed_graph = False
        self.graph = None
        if self.inputs_pl is not None:
            self.graph = self.inputs_pl.graph
            self.fixed_graph = True
        self.var_scope = None
        self.model = None
        self.trn = None

        # Outputs
        self.llikes = None
        self.sampler = None
        self.loss = None
        self.nf = None
        self.KL = None

    @property
    def sample(self):
        return self.trn.sample

    def main(self):
        self.reset_graph()
        self.build_graph()
        self.build_trainer()

        with self.graph.as_default():
            outputs = self.trn.main()

        return outputs

    def reset_graph(self):
        # TODO: make sure bandaid works
        # Set up model/trainer in graph
        if not self.fixed_graph:
            tf.reset_default_graph()
            self.graph = tf.Graph()
            self.inputs_pl = None

    def build_graph(self):
        fetchers = self.fetchers
        config = self.config
        with self.graph.as_default():
            if self.inputs_pl is None:
                # TODO: placeholder for set lengths
                if fetchers.train.ndatasets > 1:
                    print('Labeled Data')
                    # inputs_pl = tf.placeholder(
                    #     tf.float32, (None, None, fetchers.dim[0]), 'inputs')
                    # TODO: remove
                    inputs_pl = tf.placeholder(
                        tf.float32,
                        (None, fetchers.train._subsamp, fetchers.dim),
                        'inputs')
                    # Labeled data. Assumes one-hot
                    conditioning_pl = tf.placeholder(
                        tf.float32, (None, fetchers.dim[1]), 'conditioning')
                else:
                    # Unlabeled data.
                    # inputs_pl = tf.placeholder(
                    #     tf.float32, (None, None, fetchers.dim), 'inputs')
                    # TODO: remove
                    inputs_pl = tf.placeholder(
                        tf.float32,
                        (None, fetchers.train._subsamp, fetchers.dim),
                        'inputs')
                    conditioning_pl = None
                self.inputs_pl = inputs_pl
                self.conditioning_pl = conditioning_pl
            else:
                inputs_pl = self.inputs_pl
                conditioning_pl = self.conditioning_pl
            if config.dropout_keeprate_val is not None and \
                    config.dropout_keeprate_val < 1.0:
                self.config.dropout_keeprate = tf.placeholder(
                    tf.float32, [], 'dropout_keeprate')
            else:
                self.config.dropout_keeprate = None
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)

            with tf.variable_scope('iidvae', initializer=config.initializer) \
                    as self.var_scope:
                # TODO: remove when handling dynamic shapes
                n = int(inputs_pl.get_shape()[1])
                # n = tf.shape(inputs_pl)[1]
                self.nf = tf.cast(n, tf.float32)
                # Latent codes
                feat_mu, feat_lsig = encoder(
                    inputs_pl, config.embed_layers, config.embed_size,
                    const_lsig=config.const_lsig
                )
                feat_sig = tf.exp(feat_lsig)
                self.KL = 0.5*tf.reduce_sum(
                    feat_mu**2 + feat_sig**2 - 2*feat_lsig - 1.0, -1)
                std_gau = tf.random_normal(
                    (tf.shape(inputs_pl)[0], config.embed_size))
                latents = feat_sig*std_gau + feat_mu
                samp_latents = tf.random_normal(
                    (config.sample_batch_size, config.embed_size))
                # Incorporate any conditioning information
                if conditioning_pl is None:
                    conditioning = latents
                    samp_conditioning = samp_latents
                else:
                    # TODO: ASSUMES CONDITIONING_PL SAME N AS SAMPLING
                    conditioning = tf.concat(
                        (conditioning_pl, latents), -1)
                    samp_conditioning = tf.concat(
                        (conditioning_pl, samp_latents), -1)
                # Make conditioning info N x n x s
                if conditioning is not None:
                    conditioning = tf.expand_dims(conditioning, 1)
                    conditioning = tf.tile(conditioning, [1, n, 1])
                    samp_conditioning = tf.expand_dims(samp_conditioning, 1)
                    samp_conditioning = tf.tile(
                        samp_conditioning, [1, config.sample_batch_size_n, 1])
                with tf.variable_scope('model'):
                    self.model = iidmod.CondIIDModel(self.make_base_model())
                    self.llikes, self.sampler = \
                        self.model.build_graph(
                            inputs_pl, conditioning, samp_conditioning)
                    self.loss = -tf.reduce_mean(self.llikes-self.KL)
                    # TODO: keep training normalization?
                    self.loss /= self.nf

    def build_trainer(self):
        with self.graph.as_default():
            config = self.config
            with tf.variable_scope('train'):
                self.trn = trainer.RedTrainer(
                    self.fetchers, self.loss, self.inputs_pl,
                    (self.llikes-self.KL)/self.nf,  # TODO: correct?
                    conditioning_data=self.conditioning_pl,
                    batch_size=config.batch_size,
                    sess=self.sess,
                    init_lr=config.init_lr,
                    min_lr=config.min_lr,
                    lr_decay=config.lr_decay,
                    decay_interval=config.decay_interval,
                    penalty=config.penalty,
                    dropout_keeprate=self.config.dropout_keeprate,
                    dropout_keeprate_val=config.dropout_keeprate_val,
                    train_iters=config.train_iters,
                    hold_iters=config.hold_iters,
                    print_iters=config.print_iters,
                    hold_interval=config.hold_interval,
                    optimizer_class=config.optimizer_class,
                    max_grad_norm=config.max_grad_norm,
                    do_check=config.do_check,
                    momentum=config.momentum,
                    momentum_iter=config.momentum_iter,
                    pretrain_scope=config.pretrain_scope,
                    pretrain_iters=config.pretrain_iters,
                    summary_log_path=self.summary_location,
                    save_path=self.save_location,
                    sampler=self.sampler,
                    input_sample=False,  # TODO: might not be false w/ lbls
                    samp_per_cond=config.samp_per_cond,
                    nsamp=config.nsample_batches)

    def make_base_model(self, preproc_func=None):
        # Returns base TAN model
        return mod.TANModel(
            transformations=self.config.transformations,
            conditional_model=self.config.conditional_model,
            likefunc=self.config.likefunc,
            param_nlayers=self.config.param_nlayers,
            hidden_activation=self.config.hidden_activation,
            cond_param_irange=self.config.cond_param_irange,
            dropout_keep_prob=self.config.dropout_keeprate,
            nparams=self.config.nparams,
            base_distribution=self.config.base_distribution,
            sample_size=self.config.sample_batch_size,
            trans_conditioning=self.config.trans_conditioning,
            conditional_conditioning=self.config.conditional_conditioning,
            fc_conditioning=self.config.fc_conditioning,
            preproc_func=preproc_func,
        )
