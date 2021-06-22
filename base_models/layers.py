# -*- coding: utf-8 -*-

"""
 * @file    graph_layers.py
 * @author  chenye@antfin.com
 * @date    2019/8/11 1:27 PM
 * @brief   
"""

import keras
from keras.layers import *


class LayerNormalization(Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class Cross(Layer):
    def __init__(self, nb_layers, **kwargs):
        self.nb_layers = nb_layers
        super(Cross, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[0][1] * len(input_shape)
        self.W = []
        self.bias = []
        for i in range(self.nb_layers):
            self.W.append(self.add_weight(shape=[1, self.input_dim], initializer='glorot_uniform', name='w_' + str(i),
                                          trainable=True))
            self.bias.append(
                self.add_weight(shape=[1, self.input_dim], initializer='zeros', name='b_' + str(i), trainable=True))
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: None*(group_num*embedding_size) = (None,n)
        """
        inputs = Concatenate(axis=-1)(inputs)
        # cross = Lambda(lambda x: Add()([inputs * K.sum(x * self.W[i], axis=-1, keepdims=True),
        #                               self.bias[i], inputs]))(inputs)

        global cross
        for i in range(self.nb_layers):
            if i == 0:
                cross = Lambda(lambda x: Add()([inputs * K.sum(x * self.W[i], axis=-1, keepdims=True),
                                            self.bias[i], x]))(inputs)
            else:
                cross = Lambda(lambda x: Add()([inputs * K.sum(x * self.W[i], axis=-1, keepdims=True), self.bias[i], x]))(cross)

        return cross

    def compute_output_shape(self, input_shape):
        return None, self.input_dim


class FM(Layer):
    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = 1
        super(FM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: (None,1)
        """
        sum_squared_part_tmp = Add()(inputs)
        sum_squared_part = multiply([sum_squared_part_tmp, sum_squared_part_tmp])
        squared_sum_part_tmp = list(map(lambda x: multiply([x, x]), inputs))
        squared_sum_part = Add()(squared_sum_part_tmp)

        fm_emb = Lambda(lambda x: K.sum(x[0] - x[1], axis=-1, keepdims=True) * 0.5)(
            [sum_squared_part, squared_sum_part])

        return fm_emb

    def compute_output_shape(self, input_shape):
        return None, self.output_dim


class BiInteractionPooling(Layer):
    def __init__(self, **kwargs):
        super(BiInteractionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[0][1]
        super(BiInteractionPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: (None,embedding_size)
        """
        sum_squared_part_tmp = Add()(inputs)
        sum_squared_part = multiply([sum_squared_part_tmp, sum_squared_part_tmp])
        squared_sum_part_tmp = list(map(lambda x: multiply([x, x]), inputs))
        squared_sum_part = Add()(squared_sum_part_tmp)

        subtract = Lambda(lambda x: x[0] - x[1], output_shape=lambda shapes: shapes[0])
        fm_emb = Lambda(lambda x: x * 0.5)(subtract([sum_squared_part, squared_sum_part]))

        return fm_emb

    def compute_output_shape(self, input_shape):
        return None, self.output_dim


class AFM(Layer):
    def __init__(self, l1=0.01, l2=1e-4, dropout=0.2, **kwargs):
        super(AFM, self).__init__(**kwargs)
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout

    def build(self, input_shape):
        self.group_size = len(input_shape)
        self.embedding_size = input_shape[0][1]
        print("chenye**0603**embedding_size:", self.embedding_size)
        # self.attn_size = (self.embedding_size + 1) / 2   ### 原本code

        self.attn_size = (self.embedding_size + 1) // 2    ### 修改后code
        print("chenye**0603**attn_size:", self.attn_size)

        self.output_dim = 1

        self.attn = Dense(self.attn_size,
                          kernel_initializer=initializers.glorot_normal(),
                          kernel_regularizer=regularizers.l1_l2(self.l1, self.l2),
                          bias_initializer=initializers.Zeros(),
                          name='attention',
                          use_bias=True)

        self.proj_h = self.add_weight(shape=(self.attn_size, 1),
                                      initializer=initializers.glorot_normal(),
                                      regularizer=regularizers.l1_l2(self.l1, self.l2),
                                      name='projection_h')
        self.proj_p = self.add_weight(shape=(self.embedding_size, 1),
                                      initializer=initializers.glorot_normal(),
                                      name='projection_p')

        super(AFM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: (None,embedding_size)
        """

        products = []
        for i in range(self.group_size):
            for j in range(i + 1, self.group_size):
                products.append(Multiply()([inputs[i], inputs[j]]))

        # batch size * group_size(group_size-1)/2 * embedding_size
        mat = Lambda(lambda x: K.stack(x, axis=1))(products)

        # batch size * group_size(group_size-1)/2 * attn_size
        attn = self.attn(mat)
        attn = Activation('relu')(attn)
        alpha = Lambda(lambda x: K.dot(x[0], x[1]))([attn, self.proj_h])
        alpha = Lambda(lambda x: activations.softmax(x, axis=1))(alpha)

        # batch_size * embedding size
        output = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([alpha, mat])
        output = Dropout(self.dropout)(output)

        output = Lambda(lambda x: K.dot(x[0], x[1]))([output, self.proj_p])

        return output

    def compute_output_shape(self, input_shape):
        return None, self.output_dim


class CIN(Layer):
    def __init__(self, layers_dim, activation, **kwargs):
        self.nb_layers = len(layers_dim)
        self.layers_dim = layers_dim
        self.activation = activation
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = sum(self.layers_dim)
        self.m = len(input_shape)
        self.W = []

        for i in range(self.nb_layers):
            Hk = self.layers_dim[i]
            Hk_1 = self.m if i == 0 else self.layers_dim[i - 1]
            self.W.append(self.add_weight(shape=[1, self.m, Hk_1, 1, Hk], initializer='glorot_uniform',
                                          name='w_' + str(i), trainable=True))

        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: (None, sum(layers_dim)) = (None,n)
        """
        sparse_embedding = [Lambda(lambda x: K.expand_dims(x, 1))(emb) for emb in inputs]
        x0 = concatenate(sparse_embedding, axis=1)  # (None,m,d)
        xks = []

        for i in range(self.nb_layers):
            if i == 0:
                xk_1 = x0
            else:
                xk_1 = xks[i - 1]

            xkx0 = Lambda(lambda x: K.expand_dims(x, 1) * K.expand_dims(x0, 2))(xk_1)  # (None,m,Hk_1,d)
            xk = Lambda(lambda x: self.W[i] * K.expand_dims(x, 4))(xkx0)  # (None,m,Hk_1,d,Hk)
            xk = Lambda(lambda x: K.sum(K.sum(x, axis=1, keepdims=False), axis=1, keepdims=False))(xk)  # (None,d,Hk)
            xk = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(xk)  # (None,Hk,d)
            xk = Activation(self.activation)(xk)
            xks.append(xk)

        outputs = []
        for xk in xks:
            outputs.append(Lambda(lambda x: K.sum(x, axis=2, keepdims=False))(xk))  # [(None,H1),(None,H2)...]
        cin_output = concatenate(outputs, axis=1)
        return cin_output

    def compute_output_shape(self, input_shape):
        return None, self.output_dim


class AutoInteraction(Layer):
    def __init__(self, nb_layers,
                 nb_heads=4,
                 concat=False,
                 activation='relu',
                 dropout=0.1,
                 layer_norm=True,
                 kernel_initializer='he_normal',
                 bias_initializer='zero',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.concat = concat
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AutoInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_size = input_shape[0][1]
        self.group_num = len(input_shape)

        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: (None, sum(layers_dim)) = (None,n)
        """

        # batch_size * group_num * embedding_size
        x = Lambda(lambda l: K.stack(l, axis=1))(inputs)

        outputs = []
        for i in range(0, self.nb_layers):
            attn_list = []
            for head in range(0, self.nb_heads):
                qmat = TimeDistributed(Dense(self.embedding_size,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             activity_regularizer=self.activity_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             name='query_kenel_l{}_h{}'.format(i, head),
                                             use_bias=False))(x)
                kmat = TimeDistributed(Dense(self.embedding_size,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             activity_regularizer=self.activity_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             name='key_kenel_l{}_h{}'.format(i, head),
                                             use_bias=False))(x)
                vmat = TimeDistributed(Dense(self.embedding_size,
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             activity_regularizer=self.activity_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             name='value_kenel_l{}_h{}'.format(i, head),
                                             use_bias=False))(x)

                attn_weights = Lambda(lambda l:
                                      K.batch_dot(l[0], l[1], axes=[2, 2]) / np.sqrt(self.embedding_size))([qmat, kmat])
                attn_weights = Activation('softmax',
                                          name='attention_softmax_l{}_h{}'.format(i, head))(attn_weights)
                if self.dropout > 1e-4:
                    attn_weights = Dropout(self.dropout)(attn_weights)

                attn = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn_weights, vmat])
                attn_list.append(attn)

            attn = Concatenate(axis=-1)(attn_list)
            attn = TimeDistributed(Dense(self.embedding_size,
                                         kernel_initializer=self.kernel_initializer,
                                         kernel_regularizer=self.kernel_regularizer,
                                         activity_regularizer=self.activity_regularizer,
                                         kernel_constraint=self.kernel_constraint,
                                         name='attn_kenel_l{}'.format(i),
                                         use_bias=False))(attn)

            if self.dropout > 1e-4:
                attn = Dropout(self.dropout)(attn)

            if self.layer_norm:
                attn = Add()([attn, x])
                attn = LayerNormalization()(attn)

            outputs.append(attn)
            x = attn

        if self.concat:
            x = Concatenate(axis=-1)(outputs)
            x = Reshape((self.group_num * self.embedding_size * self.nb_layers,))(x)
        else:
            x = Reshape((self.group_num * self.embedding_size,))(x)

        return x

    def compute_output_shape(self, input_shape):
        if self.concat:
            return None, self.group_num * self.embedding_size * self.nb_layers
        else:
            return None, self.group_num * self.embedding_size


class AttentiveCross(Layer):
    def __init__(self, nb_layers, l1=0.01, l2=1e-4, dropout=0.1, **kwargs):
        self.nb_layers = nb_layers
        self.dropout = dropout
        self.l1 = l1
        self.l2 = l2
        super(AttentiveCross, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_size = input_shape[0][1]
        self.group_num = len(input_shape)
        self.bias = []
        self.kernels = []
        self.attn_kernels = []
        for i in range(self.nb_layers):
            self.kernels.append(
                TimeDistributed(Dense(self.embedding_size, init='he_normal', activation=None, use_bias=False,
                                      kernel_regularizer=keras.regularizers.L1L2(self.l1, self.l2)))
            )
            self.attn_kernels.append([
                TimeDistributed(Dense(1, init='he_normal', activation=None, use_bias=False,
                                      kernel_regularizer=keras.regularizers.L1L2(self.l1, self.l2))),
                TimeDistributed(Dense(1, init='he_normal', activation=None, use_bias=False,
                                      kernel_regularizer=keras.regularizers.L1L2(self.l1, self.l2)))
            ])

            self.bias.append(
                self.add_weight(shape=[1, self.group_num, self.embedding_size], initializer='zeros', name='b_' + str(i),
                                trainable=True))

        self.mask = K.expand_dims(
            K.exp((K.ones([self.group_num, self.group_num]) - K.eye(self.group_num)) * -10e9) * -10e9, 0)

        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param inputs: [(None,embedding_size),(None,embedding_size)...]
        :return: None*(group_num*embedding_size) = (None,n)
        """
        # batch_size * group_num * embedding_size
        inputs = Lambda(lambda x: K.stack(x, axis=1))(inputs)

        def attentive_cross_interaction(x0, xk, idx):
            new_x0 = self.kernels[idx](x0)
            new_xk = self.kernels[idx](xk)

            attn_x0 = self.attn_kernels[idx][0](new_x0)
            attn_xk = self.attn_kernels[idx][1](new_xk)

            # batch_size * group_num * group_num
            attn_weight = Lambda(lambda x: x[0] + K.permute_dimensions(x[1], [0, 2, 1]))([attn_x0, attn_xk])
            attn_weight = Activation('relu')(attn_weight)

            attn_weight = Add()([attn_weight, self.mask])
            attn_weight = Activation('softmax')(attn_weight)

            # batch_size * group_num * embedding_size
            vmat = Lambda(lambda x: x[0] * x[1])([new_x0, new_xk])

            if self.dropout > 1e-4:
                attn_weight = Dropout(self.dropout)(attn_weight)

            attn = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn_weight, vmat])

            return attn

        cross = Lambda(lambda x: Add()([attentive_cross_interaction(inputs, x, 0), self.bias[0], inputs]))(inputs)
        for i in range(1, self.nb_layers):
            cross = Lambda(lambda x: Add()([attentive_cross_interaction(inputs, x, i), self.bias[i], inputs]))(cross)

        cross = Reshape((self.group_num * self.embedding_size,))(cross)
        return cross

    def compute_output_shape(self, input_shape):
        return None, self.group_num * self.embedding_size
