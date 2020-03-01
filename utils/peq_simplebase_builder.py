import tensorflow as tf
from ..experiments import runner
import copy  # noqa
import numpy as np
from datetime import datetime  # noqa
from ..model import transforms as trans  # noqa
from ..experiments import simplebase_experiment as set_exp
from ..experiments import config  # noqa


def main(
    # data
    x,
    samp_size=128,
    reuse=tf.AUTO_REUSE,
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
):
    d = int(x.get_shape()[2])

    def randperm():
        return np.random.permutation(d)

    print('Permutation Equivariant Transformations w/ Simple Base Builder')
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
        set_transformations += [lambda X: trans.expsum_peq_layer(X, tau=tau), ]
    setnvp_transformations = [
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
    ac = copy.copy(runner.DEF_ARGS)
    ac['do_check'] = False
    ac['set_transformations'] = set_transformations
    ac['sample_batch_size_n'] = int(x.get_shape()[1])
    ac['sample_batch_size'] = samp_size

    config_args = config.RedConfig(**ac)
    with tf.variable_scope('set_likelihood', reuse=reuse):
        exp = set_exp.SimpleBaseExperiment(
            config_args, None, None, None, inputs_pl=x)
        exp.build_graph()
    return exp.llikes, exp.sampler
