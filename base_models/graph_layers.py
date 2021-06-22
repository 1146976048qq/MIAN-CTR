# -*- coding: utf-8 -*-

"""
 * @file    graph_layers.py
 * @author  chenye@antfin.com
 * @date    2019/8/1 12:57 PM
 * @brief   
"""

from keras.layers import *
import tensorflow as tf


class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, use_bn=True, normalize=True, use_bias=True, dropout=0.1,
                 act='linear'):
        super(GraphConvolution, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bn = use_bn
        self.normalize = normalize
        self.use_bias = use_bias
        self.dropout = dropout
        self.act = act

    def build(self, input_shape):
        self.group_num = input_shape[1][1]
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param  inputs: (x, adj)
                    x: batch_size * group num * input_dim
                    adj: batch_size * group num * group num
        :return:    x: batch_size * group num * output_dim
        """

        x, adj = inputs

        if self.dropout > 1e-4:
            x = Dropout(self.dropout)(x)

        from .utils import symmetric_normalized_laplacian

        adj = symmetric_normalized_laplacian(adj)

        y = Lambda(lambda l: K.batch_dot(l[0], l[1]))([adj, x])
        y = TimeDistributed(Dense(self.output_dim, activation=None, use_bias=self.use_bias))(y)

        if self.normalize:
            y = Lambda(lambda l: tf.nn.l2_normalize(l, axis=-1))(y)

        if self.act != 'linear':
            y = Activation(self.act)(y)

        if self.use_bn:
            y = BatchNormalization(axis=1)(y)

        return y

    def compute_output_shape(self, input_shape):
        return None, self.group_num, self.output_dim


class MeanGraphSage(Layer):
    def __init__(self, input_dim, output_dim, use_bn=True, add_loop=True, normalize=True, use_bias=True, dropout=0.1,
                 act='linear'):
        super(MeanGraphSage, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bn = use_bn
        self.add_loop = add_loop
        self.normalize = normalize
        self.use_bias = use_bias
        self.dropout = dropout
        self.act = act

    def build(self, input_shape):
        self.group_num = input_shape[1][1]
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param  inputs: (x, adj)
                    x: batch_size * group num * input_dim
                    adj: batch_size * group num * group num
        :return:    x: batch_size * group num * output_dim
        """

        x, adj = inputs

        if self.dropout > 1e-4:
            x = Dropout(self.dropout)(x)

        if self.add_loop:
            adj = Lambda(
                lambda l: tf.maximum(tf.expand_dims(tf.eye(tf.cast(l.shape[1], tf.int32), dtype=tf.float32), axis=0),
                                     l))(adj)

        adj = Lambda(lambda l: l / tf.reduce_sum(l, axis=-1, keepdims=True))(adj)
        y = Lambda(lambda l: K.batch_dot(l[0], l[1]))([adj, x])

        y = TimeDistributed(Dense(self.output_dim, activation=None, use_bias=self.use_bias))(y)

        if self.normalize:
            y = Lambda(lambda l: tf.nn.l2_normalize(l, axis=-1))(y)

        if self.act != 'linear':
            y = Activation(self.act)(y)

        if self.use_bn:
            y = BatchNormalization(axis=1)(y)

        return y

    def compute_output_shape(self, input_shape):
        return None, self.group_num, self.output_dim


def DiffPool(x, adj, s, link_pred=True):
    """
    :param  x: batch_size * group num * embedding size
    :param  adj: batch_size * group num * group num
    :param  s: batch_size * group num * cluster num
    :param link_pred: bool
    :return:    out(new_x): batch_size * cluster num * embedding size
                out_adj(new_adj): batch_size * cluster num * cluster num
                reg(link prediction loss):  batch_size * cluster num * cluster num
    """

    out = Lambda(lambda l: K.batch_dot(l[0], l[1], axes=[1, 1]))([s, x])
    out_adj = Lambda(lambda l: K.batch_dot(K.batch_dot(l[0], l[1], axes=[1, 1]), l[0]),
                     )([s, adj])

    reg = 0.
    if link_pred:
        reg = Lambda(lambda l: l[1] - K.batch_dot(l[0], l[0], axes=[2, 2]))([s, adj])
        reg = Lambda(lambda l: tf.norm(l, ord=2))(reg)
        reg = Lambda(lambda l: l[0] / tf.cast(tf.size(l[1]), tf.float32))([reg, adj])
    return out, out_adj, reg


class GNN(Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True, add_loop=False, lin=True, dropout=0.1,
                 conv='graphsage'):
        super(GNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if conv == 'graphsage':
            self.conv1 = MeanGraphSage(input_dim, hidden_dim, normalize=normalize, act='relu', add_loop=add_loop,
                                       dropout=dropout)
            self.conv2 = MeanGraphSage(hidden_dim, hidden_dim, normalize=normalize, act='relu', add_loop=add_loop,
                                       dropout=dropout)
            self.conv3 = MeanGraphSage(hidden_dim, output_dim, normalize=normalize, act='relu', add_loop=add_loop,
                                       dropout=dropout)
        elif conv == 'gcn':
            self.conv1 = GraphConvolution(input_dim, hidden_dim, normalize=normalize, act='relu', dropout=dropout)
            self.conv2 = GraphConvolution(hidden_dim, hidden_dim, normalize=normalize, act='relu', dropout=dropout)
            self.conv3 = GraphConvolution(hidden_dim, output_dim, normalize=normalize, act='relu', dropout=dropout)

        if lin:
            self.lin = TimeDistributed(Dense(output_dim, activation='relu'))
        else:
            self.lin = None

    def build(self, input_shape):
        self.group_num = input_shape[1][1]
        self.built = True

    def call(self, inputs, **kwargs):
        """
        :param  inputs: (x, adj)
                    x: batch_size * group num * input_dim
                    adj: batch_size * group num * group num
        :return:    x: batch_size * group num * 3 * output_dim or batch_size * group num * output_dim
        """

        x0, adj = inputs
        x1 = self.conv1([x0, adj])
        x2 = self.conv2([x1, adj])
        x3 = self.conv3([x2, adj])

        x = Concatenate(axis=-1)([x1, x2, x3])

        if self.lin:
            x = self.lin(x)

        return x

    def compute_output_shape(self, input_shape):
        if self.lin:
            return None, self.group_num, self.output_dim
        else:
            return None, self.group_num, 2 * self.hidden_dim + self.output_dim
