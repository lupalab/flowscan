import os
import tensorflow as tf
from ..experiments import runner
import copy  # noqa
import numpy as np
import requests
from datetime import datetime  # noqa
from ..data import pointcloud_fetcher as pfetcher
from ..model import transforms as trans  # noqa
from ..experiments import gcn_simplebase_experiment as set_exp
from ..experiments import config  # noqa
from ..utils import misc as umisc
import plot3dscatter as pscat


def main(
    # data
    home=None,
    datadir=None,
    dataset='plane',
    keep_as_is=False,
    d=3,  # TODO: get from data
    subsamp=512,
    # transformations
    use_scale=True,
    max_scale=2.0,
    hidden=[128, 128],
    ncoupling=0,
    activation=tf.nn.relu,
    use_linear_peq=True,
    # set nvp
    use_stats=True,
    setnvp_embed=256,
    setnvp_hidden=[256, 256],
    nsetnvp=8,
    setnvp_irange=1e-6,
    nsettrans=2,
    tau=0.0,  # None -> learnable inverse-temperature
    # training
    get_forward=False,
    experiment='peq_simplebase',
    ntrls=1,
    trial_iter_ratio=0.25,
    batch_size=80,
    max_grad_norm=None,
    init_lr=0.005,
    train_iters=50000,
    print_iters=100,
    decay_interval=5000,
):
    def randperm():
        return np.random.permutation(d)

    if datadir is None:
        home = os.path.expanduser('~') if home is None else home
        datadir = os.path.join(home, 'data/tan/pntcloud')
    data_path = os.path.join(datadir, dataset+'.p')
    umisc.make_path(datadir)
    savedir = os.path.join(datadir, dataset)
    umisc.make_path(savedir)
    print('Permutation Equivariant Transformations w/ Simple Base Demo')
    print('Running experiment: ' + experiment)
    # Transformations
    if max_scale is not None:
        print('Max scale:{}'.format(max_scale))
        scale_function = trans.capped_sigmoid(max_scale)
    else:
        scale_function = None
    pnt_transformations = []
    for ci in range(ncoupling):
        pnt_transformations += [
            lambda x: trans.permute(x, randperm()),
            lambda x: trans.additive_coupling(
                x, hidden_sizes=hidden, activation=activation,
                use_scale=use_scale, scale_function=scale_function),
            lambda x: trans.shift(x),
            lambda x: trans.log_rescale(x),
        ]
    if len(pnt_transformations) > 0:
        set_transformations = [
            lambda X: trans.set_pnt_trans(
                X, lambda x: trans.transformer(x, pnt_transformations)),
        ]
    else:
        set_transformations = []
    if use_linear_peq:
        # set_transformations += [lambda X: trans.linear_peq_layer(X), ]
        set_transformations += [lambda X: trans.expsum_peq_layer(X, tau=tau), ]
    # conf = config.RedConfig()
    # interpnt_transformations = [
    #     trans.linear_map,
    #     trans.leaky_transformation,
    #     lambda x: trans.rnn_coupling(x, conf.rnn_coupling_class),
    # ]
    setnvp_transformations = [
        # lambda X: trans.set_pnt_trans(
        #     X, lambda x: trans.transformer(x, interpnt_transformations)),
        lambda X: trans.set_nvp(
            X, setnvp_hidden, 2 if d == 3 else None,
            setnvp_irange, embed_size=setnvp_embed,
            use_scale=use_scale, scale_function=scale_function,
            use_stats=use_stats
        ),
        lambda X: trans.set_nvp(
            X, setnvp_hidden, 1 if d == 3 else None,
            setnvp_irange, embed_size=setnvp_embed,
            use_scale=use_scale, scale_function=scale_function,
            use_stats=use_stats
        ),
        lambda X: trans.set_nvp(
            X, setnvp_hidden, 0 if d == 3 else None,
            setnvp_irange, embed_size=setnvp_embed,
            use_scale=use_scale, scale_function=scale_function,
            use_stats=use_stats
        ),
    ]
    set_transformations += nsetnvp*setnvp_transformations
    if use_linear_peq and nsetnvp > 0:
        # set_transformations += [lambda X: trans.linear_peq_layer(X), ]
        set_transformations += [lambda X: trans.expsum_peq_layer(X, tau=tau), ]
    set_transformations = nsettrans*set_transformations
    # Options
    ac = {}
    if print_iters is not None:
        ac['print_iters'] = (print_iters, )
    ac['do_check'] = (False, )
    ac['init_lr'] = (init_lr, )
    ac['decay_interval'] = (decay_interval, )
    ac['max_grad_norm'] = (max_grad_norm, )
    ac['train_iters'] = (train_iters, )
    ac['batch_size'] = (batch_size, )
    ac['hold_iters'] = (64, )
    ac['set_transformations'] = (set_transformations, )
    ac['sample_batch_size_n'] = (subsamp, )
    ac['get_forward'] = (get_forward, )
    ac['trial'] = range(ntrls)

    ename = '{}_n_{}'.format(experiment, subsamp)
    print(ename)
    savedir = os.path.join(datadir, dataset, ename)
    umisc.make_path(savedir)

    noisestd = 0.001
    # Train
    ret = runner.run_experiment(
        data_path,
        arg_list=runner.misc.make_arguments(ac),
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
        pscat.save_scatter(samps, savedir, result_str)
        with open(os.path.join(savedir,
                               'conf_{}.txt'.format(result_str)),
                  'w') as tout:
            tout.write(result_str)
            tout.write('\n{}'.format(ac))
        np.save(
            os.path.join(
                savedir, '{}_nll-{:0.4}_samp_{}.npy'.format(
                    dataset, nll, ename)),
            samps)
        if get_forward:
            forward = ret[0]['results']['forward']
            np.save(
                os.path.join(
                    savedir, '{}_nll-{:0.4}_forward_{}.npy'.format(
                        dataset, nll, ename)),
                forward)
    return ret
