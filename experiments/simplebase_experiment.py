import tensorflow as tf
import numpy as np  # noqa
import trainer
from ..model import simplebase_model as iidbasemod


class SimpleBaseExperiment:

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
                    print('Labeled Images')
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
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)

            with tf.variable_scope('IIDBase', initializer=config.initializer) \
                    as self.var_scope:
                # TODO: remove when handling dynamic shapes
                n = int(inputs_pl.get_shape()[1])
                # n = tf.shape(inputs_pl)[1]
                self.nf = tf.cast(n, tf.float32)

                conditioning = conditioning_pl
                samp_conditioning = conditioning_pl
                # # Make conditioning info N x n x s
                if conditioning is not None:
                    conditioning = tf.expand_dims(conditioning, 1)
                    conditioning = tf.tile(conditioning, [1, n, 1])
                    samp_conditioning = tf.expand_dims(samp_conditioning, 1)
                    samp_conditioning = tf.tile(
                        samp_conditioning, [1, config.sample_batch_size_n, 1])
                with tf.variable_scope('model'):
                    if self.config.get_forward:
                        self.forward_tensors = [inputs_pl+0.0]
                    else:
                        self.forward_tensors = None
                    self.model = iidbasemod.SimpleBaseModel(
                        config.set_transformations,
                        base_distribution=config.base_distribution,
                        sample_size=self.config.sample_batch_size,
                        sample_size_n=self.config.sample_batch_size_n,
                    )
                    self.llikes, self.sampler = \
                        self.model.build_graph(
                            inputs_pl, conditioning, samp_conditioning,
                            forward_tensors=self.forward_tensors,
                        )
                    if self.config.get_forward:
                        self.forward_tensors = tf.stack(self.forward_tensors)
                    self.loss = -tf.reduce_mean(self.llikes)
                    # TODO: keep training normalization?
                    self.loss /= self.nf

    def build_trainer(self):
        with self.graph.as_default():
            config = self.config
            with tf.variable_scope('train'):
                self.trn = trainer.RedTrainer(
                    self.fetchers, self.loss, self.inputs_pl,
                    self.llikes/self.nf,
                    conditioning_data=self.conditioning_pl,
                    batch_size=config.batch_size,
                    sess=self.sess,
                    init_lr=config.init_lr,
                    min_lr=config.min_lr,
                    lr_decay=config.lr_decay,
                    decay_interval=config.decay_interval,
                    penalty=config.penalty,
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
                    forward_tensors=self.forward_tensors,
                    samp_per_cond=config.samp_per_cond,
                    nsamp=config.nsample_batches)
