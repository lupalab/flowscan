import tensorflow as tf
import numpy as np  # noqa
from . import trainer
from ..model import single_sort_model as sortmod
from ..model import condiid_model as iidmod
from ..model import corresponding_model as corrmod
from ..model import dyn_seq_model as seqmod
from ..model import hier_model as hiermod
from ..model import embedding_model as embmod
from ..model import model as mod
from ..utils import nn  # noqa
from ..rnn import cells


class SingleSortExperiment:

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
        self.nll = None
        self.llikes = None
        self.sampler = None

        self.loss = None

        self.nf = None
        self.KL = None

    @property
    def sample(self):
        return self.trn.sample

    # TODO: a good amount of this should be inherited
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
                        (None, fetchers.train._subsamp, fetchers.dim[0]),
                        'inputs')
                    # Labeled data. Assumes one-hot
                    conditioning_pl = tf.placeholder(
                        tf.float32, [None] + fetchers.dim[1], 'conditioning')
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

            with tf.variable_scope(
                    'flowscan', initializer=config.initializer) \
                    as self.var_scope:
                # TODO: remove when handling dynamic shapes
                n = int(inputs_pl.get_shape()[1])
                # n = tf.shape(inputs_pl)[1]
                nf = tf.cast(n, tf.float32)
                print('No VAE')

                # Convert conditioning information to vector (if needed)
                if conditioning_pl is not None and \
                        config.conditioning_net is 'conv':
                    # Check that the conditioning data looks like an image
                    cond_shape = conditioning_pl.shape.as_list()[1:]
                    if len(cond_shape) is 2:
                        # Give the user the benefit of the doubt
                        conditioning_pl = tf.expand_dims(conditioning_pl,
                                                         axis=-1)
                        cond_shape += [1]

                    assert len(cond_shape) is 3, "image conditionals must " \
                        "be of rank 3, not {}.".format(len(cond_shape))

                    conditioning_pl = nn.convnet(
                        conditioning_pl, name='cond_conv',
                        **config.condition_params)

                elif conditioning_pl is not None and \
                        config.conditioning_net is 'fc':
                    # Check that the conditioning data is a vector
                    cond_shape = conditioning_pl.shape.as_list()[1:]
                    assert len(cond_shape) is 1, "vector conditionals must " \
                        "be of rank 1, not {}.".format(len(cond_shape))

                    conditioning_pl = nn.fc_network(
                        conditioning_pl, name='cond_fc',
                        **config.condition_params)

                elif conditioning_pl is not None and \
                        config.conditioning_net is not None:
                    # Check that the conditioning information is already in an
                    # acceptable form and that no transformations are needed
                    raise TypeError("conditioning must be None or a "
                                    + "placeholder (is {}) and the "
                                    + "configuration conditioning net must be "
                                    + "one of 'conv', 'fc', or None (is {})"
                                    .format(conditioning_pl,
                                            config.conditioning_net))

                # Check that the conditioning_pl has been converted to a vector
                assert conditioning_pl is None or \
                    len(conditioning_pl.get_shape()[1:]) is 1, \
                    "conditioning_pl must be None or rank 1 (is rank {})" \
                    .format(len(conditioning_pl.get_shape()[1:]))

                # Copy the conditioning information for likelihood estimation
                # and sampling
                conditioning = conditioning_pl
                samp_conditioning = conditioning_pl

                self.nf = nf
                # Make conditioning info N x n x s
                if conditioning is not None:
                    conditioning = tf.expand_dims(conditioning, 1)
                    conditioning = tf.tile(conditioning, [1, n, 1])
                    samp_conditioning = tf.expand_dims(samp_conditioning, 1)
                    samp_conditioning = tf.tile(
                        samp_conditioning, [1, config.sample_batch_size_n, 1])
                with tf.variable_scope('model'):
                    if self.config.iid is True:
                        self.model = iidmod.CondIIDModel(
                            self.make_base_model())
                    elif self.config.levels is None:
                        self.model = sortmod.SingleSortModel(
                            config.set_transformations,
                            self.make_corr_model(),
                            dropout_keep_prob=config.dropout_keeprate,
                            mean_nll=False,
                        )
                    else:
                        cmodel = self.make_corr_model()
                        if self.config.embed_corr_model:
                            # Embed the corresponding model so we get global
                            # set information
                            cmodel = embmod.EmbeddingModel(
                                cmodel,
                                embed_size=self.config.embed_size,
                                embed_layers=self.config.embed_layers)

                        # Make hierarchical model
                        hmodel = hiermod.HierTANModel(
                            cmodel, level=self.config.levels)

                        self.model = sortmod.SingleSortModel(
                            config.set_transformations, hmodel,
                            dropout_keep_prob=config.dropout_keeprate,
                            mean_nll=False,
                        )
                    self.llikes, self.sampler = \
                        self.model.build_graph(
                            inputs_pl, conditioning, samp_conditioning)

                    self.loss = -tf.reduce_mean(self.llikes)
                    self.nll = self.loss
                    # TODO: keep training normalization?
                    self.loss /= self.nf

    def build_trainer(self):
        with self.graph.as_default():
            config = self.config
            with tf.variable_scope('train'):
                self.trn = trainer.RedTrainer(
                    self.fetchers, self.loss, self.inputs_pl,
                    self.llikes/self.nf,  # TODO: correct?
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

    def make_seq_model(self):
        # Returns a sequential model for N x n x d points if seq_base is True;
        # otherwise, it returns a base TAN model (that flattens things to
        # N x nd)
        if self.config.seq_base:
            vec_model = self.make_base_model()
            # TODO: make config argument
            cell_class = cells.GRUCell(
                units=self.config.seq_feats,
                num_layers=self.config.seq_cell_layers)
            # print('\n\nUSING SRU\n\n')
            # cell_class = cells.SRUCell(units=self.config.seq_feats)
            return seqmod.SeqTANModel(
                vec_model, cell_class, preproc_func=None,
                markov_feat=self.config.use_markov_feats,
                seq_feats=self.config.seq_feats,
                fc_layers=self.config.seq_fc_layers,
                sample_size_n=self.config.sample_batch_size_n,
            )
        return self.make_base_model()

    def make_corr_model(self):
        # Returns a model that uses corresponding point information through a
        # transformation of variables if num_corr_trans is not None
        # otherwise, it returns a base TAN model
        if self.config.num_corr_trans is not None:
            return corrmod.CorrespondingModel(
                self.make_seq_model(),
                irange=self.config.corr_irange,
                num_corr_trans=self.config.num_corr_trans,
                hidden_sizes=self.config.corr_hidden_sizes,
                use_scale=self.config.corr_use_scale,
                scale_function=self.config.corr_scale_function,
            )
        return self.make_seq_model()
