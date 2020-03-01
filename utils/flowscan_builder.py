import copy  # noqa
import tensorflow as tf
import matplotlib; matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
from datetime import datetime  # noqa
from ..model import transforms as trans  # noqa
from ..experiments import runner
from ..experiments import single_sort_experiment as set_exp
from ..experiments import config


def main(
    # data
    x,
    samp_size=128,
    reuse=tf.AUTO_REUSE,
    # transformations
    use_conditional_transformations=False,
    use_initpnt_transformations=True,
    use_set_transformations=False,
    use_finalpnt_transformations=False,
    use_scale=False,
    max_scale=None,
    # set nvp
    use_presort_peq=False,
    use_stats=False,
    set_nvp_embed=256,
    code_nvp_hidden=[256, 256],
    tau=None,  # None -> learnable inverse-temperature
    num_set_transformations=2,
    # scan
    use_markov_feats=True,
    nunits=256,
    seq_base=True,
    seq_feats=256,
    seq_cell_layers=2,
    # corresponding coupling
    coupling_nunits=256,
    corr_irange=1e-8,
    num_corr_trans=None,
    corr_hidden_sizes=[],
    embed_corr_model=False,  # embed the corresponding model
):
    print('FlowScan Builder')
    # Base TAN Options
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
    # Set up transformations
    if max_scale is not None:
        print('Max scale:{}'.format(max_scale))
        scale_function = trans.capped_sigmoid(max_scale)
    else:
        scale_function = None
    conf = config.RedConfig()
    pnt_transformations = [
        trans.linear_map,
        trans.leaky_transformation,
        lambda x: trans.rnn_coupling(x, conf.rnn_coupling_class),
        trans.log_rescale, trans.leaky_transformation, trans.reverse,
        lambda x: trans.rnn_coupling(x, conf.rnn_coupling_class),
        trans.log_rescale, trans.leaky_transformation, trans.reverse,
    ]
    set_pnt_transformations = [
        lambda X: trans.set_pnt_trans(
            X, lambda x: trans.transformer(x, pnt_transformations)),
    ]
    set_transformations = [
        lambda X: trans.set_nvp(
            X, [], 2, 1e-6, embed_size=set_nvp_embed, use_scale=use_scale,
            scale_function=scale_function,
        ),
        lambda X: trans.set_nvp(
            X, [], 1, 1e-6, embed_size=set_nvp_embed, use_scale=use_scale,
            scale_function=scale_function,
        ),
        lambda X: trans.set_nvp(
            X, [], 0, 1e-6, embed_size=set_nvp_embed, use_scale=use_scale,
            scale_function=scale_function,
        ),
    ]
    if use_presort_peq:
        set_transformations.append(
            lambda X: trans.expsum_peq_layer(X, irange=1e-8, tau=tau))
    set_transformations *= num_set_transformations
    if use_presort_peq:
        peq_transformations = [
            lambda X: trans.set_pnt_trans(
                X, lambda x: trans.transformer(x, [trans.log_rescale])),
            lambda X: trans.expsum_peq_layer(X, irange=1e-8, tau=tau),
            lambda X: trans.set_pnt_trans(
                X, lambda x: trans.transformer(x, [trans.log_rescale])),
        ]
        set_transformations = peq_transformations + set_transformations + \
            peq_transformations
    transformations = []
    if use_initpnt_transformations:
        transformations += set_pnt_transformations
    if use_set_transformations:
        transformations += set_transformations
    if use_finalpnt_transformations and use_conditional_transformations:
        transformations += [
            lambda X, c: trans.set_pnt_trans(
                X,
                lambda x, cond: trans.conditioning_transformation(
                    x, cond, code_nvp_hidden, 1e-6, 1e-6, use_scale=use_scale,
                    scale_function=scale_function,
                ),
                c),
        ]
    transformations = None if len(transformations) == 0 else transformations
    # Options
    ac['samp_per_cond'] = 1
    # Number of layers to create a FC network to pass hidden states through
    ac['set_transformations'] = transformations  # NOTE: these are presort
    ac['use_markov_feats'] = use_markov_feats
    ac['seq_base'] = seq_base
    ac['seq_feats'] = seq_feats
    ac['seq_cell_layers'] = seq_cell_layers
    ac['levels'] = None
    ac['do_check'] = False
    ac['sample_batch_size_n'] = int(x.get_shape()[1])
    ac['sample_batch_size'] = samp_size
    print('set_trans:\n{}'.format(transformations))
    ac['corr_irange'] = corr_irange
    ac['num_corr_trans'] = num_corr_trans
    ac['corr_hidden_sizes'] = corr_hidden_sizes
    ac['corr_use_scale'] = use_scale
    ac['corr_scale_function'] = scale_function
    ac['embed_corr_model'] = embed_corr_model
    # TODO: coment out below?
    ac['rnn_params'] = {'nunits': nunits, 'num_layers': 1}
    ac['rnn_coupling_params'] = {'nunits': coupling_nunits, 'num_layers': 1}

    config_args = config.RedConfig(**ac)
    with tf.variable_scope('set_likelihood', reuse=reuse):
        exp = set_exp.SingleSortExperiment(
            config_args, None, None, None, inputs_pl=x)
        exp.build_graph()
    return exp.llikes, exp.sampler
