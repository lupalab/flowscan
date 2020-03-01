import copy  # noqa
import tensorflow as tf
from ..model import transforms as trans
from ..experiments import iidvae_experiment as set_exp
from ..experiments import runner
from ..experiments import config


def main(
    # data
    x,
    samp_size=128,
    reuse=tf.AUTO_REUSE,
    # VAE
    code_size=32,
    code_embed=256,
    const_lsig=False,
    conditional_embed=128,
    conditional_hidden=[256, 256],
):
    print('Neural Statistician Builder')
    # Base Conditional Model Options
    ac = copy.copy(runner.DEF_ARGS)
    ac['lr_decay'] = 0.5
    ac['first_trainable_A'] = True
    ac['trans_funcs'] = [
        trans.leaky_transformation,
        trans.log_rescale, trans.rnn_coupling, trans.reverse,
        trans.linear_map, trans.leaky_transformation,
        trans.log_rescale, trans.rnn_coupling, trans.reverse,
        trans.linear_map, trans.leaky_transformation,
        trans.log_rescale, trans.rnn_coupling, trans.reverse,
        trans.linear_map, trans.leaky_transformation,
        trans.log_rescale, trans.rnn_coupling, trans.reverse,
        trans.linear_map, trans.leaky_transformation,
        trans.log_rescale, ]
    ac['cond_func'] = runner.conds.rnn_model
    ac['param_nlayers'] = 2
    ac['relu_alpha'] = None
    ac['do_init_cond_trans'] = True
    ac['do_final_cond_trans'] = True
    ac['conditional_conditioning'] = True
    # VAE Options
    ac['embed_size'] = code_size
    ac['embed_layers'] = code_embed
    ac['conditional_embed'] = conditional_embed
    ac['conditional_hidden'] = conditional_hidden
    ac['const_lsig'] = const_lsig
    # Additional Options
    ac['do_check'] = False
    ac['sample_batch_size_n'] = int(x.get_shape()[1])
    ac['sample_batch_size'] = samp_size

    config_args = config.RedConfig(**ac)
    with tf.variable_scope('set_likelihood', reuse=reuse):
        exp = set_exp.VAEExperiment(config_args, None, None, None, inputs_pl=x)
        exp.build_graph()
    return exp.llikes, exp.sampler
