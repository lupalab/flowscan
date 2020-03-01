import tensorflow as tf
from . import model as mod


class HierTANModel(mod.Model):

    def __init__(self, base_model, level=3, name='hier_tan'):
        self.base_model = base_model
        self.level = level
        self.name = name

    def build_graph(
        self, inputs, conditioning=None, sampler_conditioning=None,
        level=None
    ):
        """ Builds the graph for a structured sequential TAN.
        Args:
            inputs: N x n x d real tensor of batch of N sequences each with
                n points.
            conditioning: N x s or N x n x s tensor of conditioning values
        """
        # def last_dim_is_None(x):
        #     if x is None:
        #         return False
        #     return x.get_shape().as_list()[-1] is None
        # TODO: for now assume that n is 2^k and two halves are same size?
        # TODO: change down the line (check if n is odd in tf)
        if level is None:
            level = self.level
        print('---\nHierachal model, level {}'
              '\ninputs:\n{}\ncond:\n{}\nsamp_cond:\n{}\n'.format(
                  level, inputs, conditioning, sampler_conditioning))
        # Base case
        if level == 0:
            # TODO: is name ok (or add _1)?
            with tf.variable_scope('{}_level_{}'.format(self.name, level)):
                # Do not split, apply sequential model to inputs
                return self.base_model.build_graph(
                    inputs, conditioning, sampler_conditioning)

        # Make conditioning information rank 3 by tiling if needed.
        inputs_shape = tf.shape(inputs)
        n = inputs_shape[1]
        if conditioning is not None:
            cshape = tf.shape(conditioning)
            cshape_list = conditioning.get_shape().as_list()
            rank_2_cond = (len(cshape_list) == 2)
            cond_dim = cshape_list[-1]
            if rank_2_cond:
                conditioning = tf.tile(
                    tf.expand_dims(conditioning, 1), [1, n, 1])
        # External conditioning values for sampling.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            assert conditioning is not None  # Need to also train conditioning.
            sshape = tf.shape(sampler_conditioning)
            sshape_list = sampler_conditioning.get_shape().as_list()
            if rank_2_cond:
                sampler_conditioning = tf.tile(
                    tf.expand_dims(sampler_conditioning, 1), [1, n, 1])

        # Get likelihood on first half.
        inputs1 = inputs[:, ::2, ...]
        if conditioning is not None:
            if rank_2_cond:
                conditioning1 = conditioning[:, ::2, ...]
                sampler_conditioning1 = sampler_conditioning[:, ::2, ...]
            else:
                # TODO: would break if n is not even?
                # TODO: assumes n is know
                conditioning1 = tf.reshape(
                    conditioning,
                    tf.stack([cshape[0], cshape_list[1]/2, 2*cond_dim]))
                sampler_conditioning1 = tf.reshape(
                    sampler_conditioning,
                    tf.stack([sshape[0], sshape_list[1]/2, 2*cond_dim]))
        else:
            conditioning1 = None
            sampler_conditioning1 = None
        with tf.variable_scope('{}_1_level_{}'.format(self.name, level)):
            llikes1, sampler1 = self.build_graph(
                inputs1, conditioning1, sampler_conditioning1, level=level-1)

        # Get likelihood on second half conditioned on first half.
        inputs2 = inputs[:, 1::2, ...]
        if conditioning is not None:
            # concat first half conditioning information and points
            conditioning2 = tf.concat((conditioning1, inputs1), -1)
            sampler_conditioning2 = tf.concat(
                (sampler_conditioning1, sampler1), -1)
        else:
            conditioning2 = inputs1
            sampler_conditioning2 = sampler1
        # TODO: just apply basemodel instead?
        with tf.variable_scope('{}_2_level_{}'.format(self.name, level)):
            out = self.build_graph(
                inputs2, conditioning2, sampler_conditioning2, level=level-1)
            if len(out) == 3:
                _, llikes2, sampler2 = out
            elif len(out) == 2:
                llikes2, sampler2 = out
            else:
                raise ValueError

        # Put together
        llikes = llikes1 + llikes2
        sampler_shape = sampler1.get_shape().as_list()
        sampler_shape[1] = sampler_shape[1]*2
        sampler = tf.reshape(
            tf.concat((sampler1, sampler2), -1), sampler_shape)
        return llikes, sampler
