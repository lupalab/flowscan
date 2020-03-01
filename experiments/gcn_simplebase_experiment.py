import tensorflow as tf
import numpy as np  # noqa
import trainer
from ..model import simplebase_model as iidbasemod
from ..utils import gscores


class SingleSortExperiment:

    def __init__(self, config, summary_location, save_location, fetchers):
        self.config = config
        self.summary_location = summary_location
        self.save_location = save_location
        self.fetchers = fetchers

        # TF control
        self.sess = None
        self.graph = None
        self.var_scope = None
        self.model = None
        self.trn = None
        # Input placeholders
        self.inputs_pl = None
        self.conditioning_pl = None
        # Outputs
        self.llikes = None
        self.sampler = None
        self.loss = None
        self.nf = None

    @property
    def sample(self):
        return self.trn.sample

    def main(self):
        self.build_graph()
        self.build_trainer()
        with self.graph.as_default():
            outputs = self.trn.main()
        return outputs

    def build_graph(self):
        fetchers = self.fetchers
        config = self.config
        # Set up model/trainer in graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # TODO: placeholder for set lengths
            if fetchers.train.ndatasets > 1:
                print('Labeled Images')
                # inputs_pl = tf.placeholder(
                #     tf.float32, (None, None, fetchers.dim[0]), 'inputs')
                # TODO: remove
                inputs_pl = tf.placeholder(
                    tf.float32,
                    (None, fetchers.train._subsamp, fetchers.dim), 'inputs')
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
                    (None, fetchers.train._subsamp, fetchers.dim), 'inputs')
                conditioning_pl = None
            self.inputs_pl = inputs_pl
            self.conditioning_pl = conditioning_pl
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
                    self.loss /= self.nf
                    # print('\n\nOnly Log Like\n\n')
                    gloss1 = tf.map_fn(
                        lambda z_tensor: gscores.gaussian_score(
                            z_tensor,
                            do_ztransform=False, do_diagonal=False,
                            epsilon=0.0, D=200, sigma=0.5,
                            use_multid_mme=False, gpen=1.0,
                            n_hsic=16, gaussian_hsic=True, hpen=1.0)[0],
                        self.model.z)
                    # gloss2 = tf.map_fn(
                    #     lambda z_tensor: gscores.gaussian_mmd_loss(
                    #         z_tensor, D=500, sigma=0.25, frequencies=None,
                    #         random_projection=False),
                    #     self.model.z)
                    # self.loss += 100.0*tf.reduce_mean(gloss1 + gloss2)
                    # print('\n\nUsing gscore+ single MME (lambda 100) \n\n')
                    self.loss += 1.0*tf.reduce_mean(gloss1)
                    print('\n\nUsing gscore (lambda 1) \n\n')

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
