import tensorflow as tf
import likelihoods as likes
import conditionals as conds
from ..utils import set as us
from ..utils import nn


# TODO: this is an unholy behemoth that needs to be chopped up and refractored.
def set_conditional(
    inputs,
    # Corresponding conditioning set and features
    conditioning_set=None, set_embed_func=lambda x: tf.reduce_mean(x, 1),
    conditioning_set_features=None,
    # Mixture model hyper-parameters
    K=1, nparams=120, base_distribution='gaussian', use_ind_comp=False,
    # Network arguments to produce mixture model params
    hidden_sizes=[], activation=tf.nn.relu,
    # Scope
    name='set_conditional',
    **kwargs
):
    """
    Computes likelihood on input set X={x_i} given side information set Y={y_i}
    with correspondence between x_i and y_i:
            p(X | Y) = log[sum_k w_k(g(Y)) p_k(X | Y)]
        where
            p_k(X | Y) = prod_i p(x_i | f_k(g({y_j}_j), y_i))
    for a mini-batch of N such sets. Here f_k is a function that determines the
    parameters of a mixture model to model x_i (there are K such functions,
    hence a mixture of mixtures), g(Y) is a permutation invariant function,
    and w_k are weights to the mixtures. In the case where there is no Y given
    (Y={}), we take f_k(..) = theta_k, a tunable variable. If there is a given
    Y, f_k(..) = A_k f(g(Y), y_i) + b_k for a shared fully connected network f.
    Args:
        inputs: N x n x 1 real tensor of N sets, inputs[s, :, :] is a set {x_i}
        conditioning_set: N x n x d real tensors of N conditioning sets,
            conditioning_set[s, :, :] is a set {y_j}
        set_embed_func: function that takes in N x n x d conditioning set to
            N x p tensor of set embeddings
        conditioning_set_features: N x s real tensor of additional side
            information of sets X to condition on. That is, we condition using
            g'(Y) = (g(Y), set_feats).
        K: int scalar indicating how many mixture models to use.
        nparams: int scalar 3*number_of_components per mixture model.
        base_diistribution: string for the distribution of components in
            mixture model.
        use_ind_comp: Boolean use an additional component per mixture that is
            independant of conditioning information (may help numerical
            stability).
        hidden_sizes: list of ints for the hidden layer sizes of f_k's
        activation: function to use as nonlinearity in f_k's
    Returns:
        log_likelihoods: length N real tensore of conditional log likelihood
            per set
        sampler: function to sample from conditionals for specified
            sample sizes and conditioning set
    """
    print("Conditioning 1st Ignoring {}".format(kwargs))
    # Get likelihood of points based on previous dimensions and set.
    with tf.variable_scope(name) as scope:
        N = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]
        if conditioning_set is not None:
            # Embed conditioning set (permutation invariant).
            with tf.variable_scope('set_embed'):
                set_embedding = set_embed_func(conditioning_set)  # N x p
                # Add set feature side information to condition on
                if conditioning_set_features is not None:
                    set_embedding = tf.concat(
                        (set_embedding, conditioning_set_features), -1)
            # Replicate (tile) to match up with corresponding input point.
            conditioning_pnts = us.repeat_set_features(
                conditioning_set, set_embedding)  # N x n x p+d
            d_p = int(conditioning_pnts.get_shape()[-1])
            # Make into Nn x p+d
            conditioning_pnts = tf.reshape(conditioning_pnts, (-1, d_p))
            # Get parameters for mixture of mixture
            # TODO: share more/less params?
            mixofmix_params = nn.fc_network(
                conditioning_pnts, K*nparams,
                hidden_sizes, name='params_fc_net',
                activation=activation)  # Nn x K*nparams
            mixofmix_params = tf.reshape(
                mixofmix_params, (-1, n, K, nparams))  # N x n x K x params
            mixofmix_weights = nn.fc_network(
                set_embedding, K,
                hidden_sizes, name='weights_fc_net',
                activation=activation)  # N x K
            if use_ind_comp:
                ind_comp = tf.get_variable(
                    'ind_comps', dtype=tf.float32, shape=(1, 1, K, 3))
                ind_comp_tile = tf.tile(ind_comp, (N, n, 1, 1))
                mixofmix_params = tf.concat(
                    (mixofmix_params, ind_comp_tile), -1)
        elif conditioning_set_features is not None:  # TODO: check
            set_embedding = conditioning_set_features
            mixofmix_params = nn.fc_network(
                set_embedding, K*nparams, hidden_sizes, name='params_fc_net',
                activation=activation)  # N x K*nparams
            mixofmix_params = tf.reshape(
                mixofmix_params, (-1, 1, K, nparams))  # N x 1 x K x params
            mixofmix_weights = nn.fc_network(
                set_embedding, K,
                hidden_sizes, name='weights_fc_net',
                activation=activation)  # N x K
            if use_ind_comp:
                ind_comp = tf.get_variable(
                    'ind_comps', dtype=tf.float32, shape=(1, 1, K, 3))
                ind_comp_tile = tf.tile(ind_comp, (N, 1, 1, 1))
                mixofmix_params = tf.concat(
                    (mixofmix_params, ind_comp_tile), -1)
        else:
            mixofmix_params = tf.get_variable(
                'params', dtype=tf.float32, shape=(1, 1,  K, nparams))
            mixofmix_weights = tf.get_variable(
                'weights', dtype=tf.float32, shape=(1, K))

        # Get likelihoods for each mixture
        mix_pnts_lls_list = [None for _ in range(K)]
        for k in range(K):
            mparams = mixofmix_params[:, :, k, :]  # {N,1} x {n,1} x params
            # likelihoods are N x n
            mix_pnts_lls_list[k] = \
                likes.mixture_likelihoods(mparams, inputs, base_distribution)
        # Get likelihood for each set (for each mixture)
        mix_pnts_lls = tf.stack(mix_pnts_lls_list, -1)  # N x n x K
        mix_lls = tf.reduce_sum(mix_pnts_lls, 1)  # N x k
        # Get mixture of micture likelihoods
        log_exp_terms = mix_lls + mixofmix_weights
        log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - \
            tf.reduce_logsumexp(mixofmix_weights, -1)

    # Sample based on same set embedding
    def sampler(batch_size, sample_n=1000, base_distribution='gaussian',
                sampler_conditioning_set=None,
                sampler_conditioning_set_features=None):
        with tf.variable_scope(scope, reuse=True):
            if sampler_conditioning_set is not None:
                with tf.variable_scope('set_embed'):
                    set_embedding = set_embed_func(
                        sampler_conditioning_set)
                    if sampler_conditioning_set_features is not None:
                        set_embedding = tf.concat(
                            (set_embedding, sampler_conditioning_set_features),
                            -1)

                conditioning_pnts = us.repeat_set_features(
                    sampler_conditioning_set, set_embedding)
                d_p = int(conditioning_pnts.get_shape()[-1])
                conditioning_pnts = tf.reshape(
                    conditioning_pnts, (-1, d_p))

                mixofmix_params = nn.fc_network(
                    conditioning_pnts, K*nparams,
                    hidden_sizes, name='params_fc_net',
                    activation=activation)  # Nn x nparams
                mixofmix_params = tf.reshape(  # N x n x K x params
                    mixofmix_params, (-1, sample_n, K, nparams))
                mixofmix_weights = nn.fc_network(
                    set_embedding, K,
                    hidden_sizes, name='weights_fc_net',
                    activation=activation)  # N x K
                # TODO: avoid during sampling?
                if use_ind_comp:
                    ind_comp = tf.get_variable(
                        'ind_comps', dtype=tf.float32, shape=(1, 1, K, 3))
                    ind_comp_tile = tf.tile(
                        ind_comp, (batch_size, sample_n, 1, 1))
                    mixofmix_params = tf.concat(
                        (mixofmix_params, ind_comp_tile), -1)
            elif sampler_conditioning_set_features is not None:  # TODO: check
                set_embedding = sampler_conditioning_set_features
                mixofmix_params = nn.fc_network(
                    set_embedding, K*nparams, hidden_sizes,
                    name='params_fc_net',
                    activation=activation)  # N x K*nparams
                mixofmix_params = tf.reshape(
                    mixofmix_params, (-1, 1, K, nparams))  # N x 1 x K x params
                mixofmix_weights = nn.fc_network(
                    set_embedding, K,
                    hidden_sizes, name='weights_fc_net',
                    activation=activation)  # N x K
                if use_ind_comp:
                    ind_comp = tf.get_variable(
                        'ind_comps', dtype=tf.float32, shape=(1, 1, K, 3))
                    ind_comp_tile = tf.tile(ind_comp, (batch_size, 1, 1, 1))
                    mixofmix_params = tf.concat(
                        (mixofmix_params, ind_comp_tile), -1)
            else:
                mixofmix_params = tf.get_variable(
                    'params', dtype=tf.float32, shape=(1, 1,  K, nparams))
                mixofmix_weights = tf.get_variable(
                    'weights', dtype=tf.float32, shape=(1, K))

            if int(mixofmix_weights.get_shape()[0]) == 1:
                js = tf.multinomial(  # int64 1 x N
                    mixofmix_weights, batch_size, name='js')
                setparams = tf.gather(  # N x params
                    mixofmix_params[0, 0, :, :], tf.squeeze(js, 0))
                setparams = tf.tile(
                    tf.expand_dims(setparams, 1), [1, sample_n, 1])
            else:
                js = tf.multinomial(  # int64 N x 1
                    mixofmix_weights, 1, name='js')
                inds = tf.concat(  # N x 2
                    (tf.expand_dims(
                        tf.range(batch_size, dtype=tf.int64), -1), js),
                    1, name='inds')
                setparams = tf.transpose(
                    mixofmix_params, [0, 2, 1, 3])  # N x K x {n, 1} x params
                setparams = tf.gather_nd(setparams, inds)  # N x {n, 1} x p
                # When sampling based conditioned set features but not set
                # points
                if int(setparams.get_shape()[1]) == 1:
                    setparams = tf.tile(setparams, [1, sample_n, 1])
            # TODO: last dim par+3 depending on use_ind_comp, use -1?
            setparams = tf.reshape(
                setparams, (batch_size*sample_n, -1))
            samp_pnts = conds.sample_mm(setparams, base_distribution)
            samp = tf.reshape(samp_pnts, (batch_size, sample_n, 1))
        return samp

    return log_likelihoods, sampler
