import os
import tensorflow as tf
from ..experiments import runner
import numpy as np
from ..data import pointcloud_fetcher as pfetcher
from ..model import transforms as trans
from ..experiments import iidvae_experiment as set_exp
from ..utils import misc as umisc
import plot3dscatter as pscat


def main(
    # data
    home=None,
    datadir=None,
    dataset='plane',
    d=3,  # TODO: get from data
    subsamp=512,
    keep_as_is=False,
    # VAE
    code_size=32,
    code_embed=256,
    const_lsig=False,
    conditional_embed=128,
    conditional_hidden=[256, 256],
    # training
    experiment='vae',
    ntrls=1,
    trial_iter_ratio=0.25,
    batch_size=80,
    max_grad_norm=None,
    init_lr=0.005,
    train_iters=50000,
    print_iters=100,
    decay_interval=5000,
):
    if datadir is None:
        home = os.path.expanduser('~') if home is None else home
        datadir = os.path.join(home, 'data/tan/pntcloud')
    data_path = os.path.join(datadir, dataset+'.p')
    umisc.make_path(datadir)
    savedir = os.path.join(datadir, dataset)
    umisc.make_path(savedir)
    print('Neural Statistician Demo')
    print('Running experiment: ' + experiment)
    # Base Conditional Model Options
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
    # VAE Options
    ac['embed_size'] = (code_size, )
    ac['embed_layers'] = (code_embed, )
    ac['conditional_embed'] = (conditional_embed, )
    ac['conditional_hidden'] = (conditional_hidden, )
    ac['const_lsig'] = (const_lsig, )
    # Additional Options
    if print_iters is not None:
        ac['print_iters'] = (print_iters, )
    ac['do_check'] = (False, )
    ac['init_lr'] = (init_lr, )
    ac['decay_interval'] = (decay_interval, )
    ac['max_grad_norm'] = (max_grad_norm, )
    ac['train_iters'] = (train_iters, )
    ac['batch_size'] = (batch_size, )
    ac['hold_iters'] = (64, )
    ac['sample_batch_size_n'] = (subsamp, )
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
        exp_class=set_exp.VAEExperiment,
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
