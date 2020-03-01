import os
import numpy as np
import copy  # noqa
import matplotlib
import matplotlib.pyplot as plt  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
from datetime import datetime  # noqa
import tensorflow as tf

from ..data import pointcloud_fetcher as pfetcher
from ..experiments import single_sort_experiment as set_exp
from ..experiments import runner
from ..experiments import config
from ..model import transforms as trans  # noqa
from ..utils import misc as umisc
from ..utils.nn import get_default_condition_params, \
    append_default_condition_params
from . import plot3dscatter as pscat

matplotlib.use('Agg')  # noqa


def main(
    # data
    home=None,
    datadir=None,
    dataset='plane',
    d=3,  # TODO: get from data
    subsamp=512,
    keep_as_is=False,
    # transformations
    use_conditional_transformations=False,
    use_initpnt_transformations=True,
    use_set_transformations=False,
    use_finalpnt_transformations=False,
    use_scale=True,
    max_scale=2.0,
    # set nvp
    use_presort_peq=False,
    use_stats=False,
    set_nvp_embed=256,
    code_nvp_hidden=[256, 256],
    tau=None,  # None -> learnable inverse-temperature
    num_set_transformations=2,
    # scan
    levels=None,
    use_markov_feats=True,
    nunits=256,
    seq_base=True,
    seq_feats=256,
    seq_cell_layers=2,
    # corresponding coupling
    coupling_nunits=256,
    corr_irange=1e-8,
    num_corr_trans=4,
    corr_hidden_sizes=[],
    embed_corr_model=False,  # embed the corresponding model
    # training
    experiment='flowscan',
    ntrls=1,
    trial_iter_ratio=0.004,
    batch_size=100,
    max_grad_norm=100.0,
    init_lr=0.0035,
    train_iters=40000,
    print_iters=100,
    decay_interval=5000,
    use_rmsprop=False,
    cond_type='vector',
    cond_dict='default'
):
    if datadir is None:
        home = os.path.expanduser('~') if home is None else home
        datadir = os.path.join(home, 'data/tan/pntcloud')
    data_path = os.path.join(datadir, dataset+'.p')
    umisc.make_path(datadir)
    savedir = os.path.join(datadir, dataset)
    umisc.make_path(savedir)
    print('FlowScan Demo')
    print('Running experiment: ' + experiment)
    print('Loading test data for plotting.')
    # Base TAN Options
    ac = {
        'lr_decay': (0.5, ),
        'first_trainable_A': (True, ),
        'trans_funcs': (
            [trans.leaky_transformation,
                trans.log_rescale, trans.rnn_coupling, trans.reverse,
                trans.linear_map, trans.leaky_transformation,
                trans.log_rescale, trans.rnn_coupling, trans.reverse,
                trans.linear_map, trans.leaky_transformation,
                trans.log_rescale, trans.rnn_coupling, trans.reverse,
                trans.linear_map, trans.leaky_transformation,
                trans.log_rescale, trans.rnn_coupling, trans.reverse,
                trans.linear_map, trans.leaky_transformation,
                trans.log_rescale, ], ),
        'cond_func': (runner.conds.rnn_model, ),
        'param_nlayers': (2, ),
        'relu_alpha': (None, ),
        'do_init_cond_trans': (True, ),
        'do_final_cond_trans': (True, ),
        'conditional_conditioning': (True, ),
    }
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
    if use_conditional_transformations:
        set_pnt_transformations = [
            lambda X, c: trans.set_pnt_trans(
                X,
                lambda x, cond: trans.conditioning_transformation(
                    x, cond, code_nvp_hidden, 1e-6, 1e-6, use_scale=use_scale,
                    scale_function=scale_function,
                ),
                c),
            lambda X: trans.set_pnt_trans(
                X, lambda x: trans.transformer(x, pnt_transformations)),
            lambda X, c: trans.set_pnt_trans(
                X,
                lambda x, cond: trans.conditioning_transformation(
                    x, cond, code_nvp_hidden, 1e-6, 1e-6, use_scale=use_scale,
                    scale_function=scale_function,
                ),
                c),
        ]
        if cond_dict is 'default':
            cond_net, cond_dict = get_default_condition_params(cond_type)
        else:
            cond_net, cond_dict = append_default_condition_params(cond_type,
                                                                  cond_dict)

        ac['conditioning_net'] = (cond_net,)
        ac['condition_params'] = (cond_dict,)
    else:
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
    ac['samp_per_cond'] = (1, )
    if print_iters is not None:
        ac['print_iters'] = (print_iters, )
    ac['init_lr'] = (init_lr, )
    ac['decay_interval'] = (decay_interval, )
    ac['max_grad_norm'] = (max_grad_norm, )
    ac['train_iters'] = (train_iters, )
    ac['batch_size'] = (batch_size, )
    ac['hold_iters'] = (64, )
    # Number of layers to create a FC network to pass hidden states through
    ac['set_transformations'] = (transformations, )  # NOTE: these are presort
    ac['use_markov_feats'] = (use_markov_feats, )
    ac['seq_base'] = (seq_base, )
    ac['seq_feats'] = (seq_feats, )
    ac['seq_cell_layers'] = (seq_cell_layers, )
    ac['levels'] = (levels, )
    ac['do_check'] = (False, )
    if levels is None:
        ac['sample_batch_size_n'] = (subsamp, )
    else:
        ac['sample_batch_size_n'] = (subsamp/2**levels, )
    print('set_trans:\n{}'.format(transformations))
    ac['trial'] = range(ntrls)
    ac['corr_irange'] = (corr_irange, )
    ac['num_corr_trans'] = (num_corr_trans, )
    ac['corr_hidden_sizes'] = (corr_hidden_sizes, )
    ac['corr_use_scale'] = (use_scale, )
    ac['corr_scale_function'] = (scale_function, )
    ac['embed_corr_model'] = (embed_corr_model, )
    # TODO: coment out below?
    ac['rnn_params'] = ({'nunits': nunits, 'num_layers': 1}, )
    ac['rnn_coupling_params'] = (
        {'nunits': coupling_nunits, 'num_layers': 1}, )
    if use_rmsprop:
        ac['optimizer_class'] = (tf.train.RMSPropOptimizer, )
    else:
        ac['optimizer_class'] = (tf.train.AdamOptimizer, )

    ename = '{}_n_{}'.format(experiment, subsamp)
    print(ename)
    savedir = os.path.join(datadir, dataset, ename)
    umisc.make_path(savedir)

    noisestd = 0.001
    # Train
    ret = runner.run_experiment(
        data_path,
        arg_list=umisc.make_arguments(ac),
        exp_class=set_exp.SingleSortExperiment,
        fetcher_class=pfetcher.generate_fetchers(
            subsamp=subsamp, subsamp_test=subsamp, subsamp_valid=subsamp,
            unit_scale=True,
            noisestd=noisestd, keep_as_is=keep_as_is),
        home=home,
        no_log=False,  # TODO: logging hangs?
        restore_best=ntrls > 1,
        trial_iter_ratio=trial_iter_ratio
    )

    # Plot
    print('Plotting samples.')

    if ret[0] is not None:
        samps = ret[0]['results']['samples']
        nll = -np.mean(ret[0]['results']['test_llks'])
        result_str = 'nll-{:0.4}_{}_{}'.format(nll, ename, dataset)
        result_str = result_str[:200]
        if d == 3:
            pscat.save_scatter(samps, savedir, result_str)
        with open(os.path.join(savedir, 'conf_{}.txt'.format(result_str)),
                  'w') as tout:
            tout.write(result_str)
            tout.write('\n{}'.format(ac))
        np.save(
            os.path.join(
                savedir, '{}_nll-{:0.4}_samp_{}.npy'.format(
                    dataset, nll, ename)),
            samps)
    return ret
