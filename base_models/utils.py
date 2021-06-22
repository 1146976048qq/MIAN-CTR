from __future__ import absolute_import

import tensorflow as tf
from keras.layers import *


def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def build_addon_branch(model, inputs, args):
    from layers import Cross, FM, AFM, CIN, AutoInteraction, AttentiveCross, BiInteractionPooling
    if model == 'dfm':
        addon_branch = FM()(inputs)
    elif model == 'nfm':
        addon_branch = BiInteractionPooling()(inputs)
    elif model == 'afm':
        dropout = args.get('dropout', 0.2)
        l1 = args.get('l1', 0.01)
        l2 = args.get('l2', 1e-4)
        addon_branch = AFM(l1=l1, l2=l2, dropout=dropout)(inputs)
    elif model == 'dcn':
        nb_layers = args.get('nb_layers', 4)
        addon_branch = Cross(nb_layers=nb_layers)(inputs)
    elif model == 'xdfm':
        layers_dim = args.get('layers_dim', [8, 8, 8, 8])
        activation = args.get('activation', 'linear')
        addon_branch = CIN(layers_dim=layers_dim, activation=activation)(inputs)
    elif model == 'autoint':
        nb_layers = args.get('layers_dim', 3)
        nb_heads = args.get('nb_heads', 4)
        concat = args.get('concat', True)
        dropout = args.get('dropout', 0.2)
        layer_norm = args.get('layer_norm', True)
        addon_branch = AutoInteraction(nb_layers=nb_layers, nb_heads=nb_heads, concat=concat, dropout=dropout,
                                       layer_norm=layer_norm)(inputs)
    elif model == 'xdcn':
        nb_layers = args.get('nb_layers', 4)
        dropout = args.get('dropout', 0.2)
        l1 = args.get('l1', 0.01)
        l2 = args.get('l2', 1e-4)
        addon_branch = AttentiveCross(nb_layers=nb_layers, dropout=dropout, l1=l1, l2=l2)(inputs)
    else:
        raise 'model %s has not been implemented' % model

    return addon_branch


def dense2sparse(dense_tensor, max_id, col_num, convert2keras=True):
    '''
    split batch_size * col_num matrix to SparseTensor list of length col_num
    '''
    dense_tensor = tf.cast(dense_tensor, tf.int64)
    dense_tensor = tf.mod(dense_tensor, max_id)
    V = tf.split(dense_tensor, col_num, axis=1)
    row_num = tf.shape(dense_tensor, out_type=tf.int64)[0]
    index = tf.expand_dims(tf.range(0, row_num, dtype=tf.int64), axis=1)
    dense_shape = [row_num, max_id]
    sps = []
    for v in V:
        indices = tf.concat([index, v], axis=1)
        values = tf.reshape(v, [-1])
        sp = tf.SparseTensor(indices=indices, values=values,
                             dense_shape=dense_shape)
        if convert2keras:
            sp = Input(sparse=True, tensor=sp, shape=(max_id,))
        sps.append(sp)
    return sps


def symmetric_normalized_laplacian(adj):
    adj = tf.maximum(tf.expand_dims(tf.eye(tf.cast(adj.shape[1], tf.int32), dtype=tf.float32), axis=0), adj)
    row_sum = tf.reduce_sum(adj, axis=2)
    d_inv_sqrt = tf.pow(row_sum, -0.5)
    d_inv_sqrt = tf.reshape(d_inv_sqrt, [-1, adj.shape[1]])
    d_mat_inv_sqrt = tf.matrix_diag(d_inv_sqrt)

    output = tf.matmul(d_mat_inv_sqrt, adj)
    output = tf.matmul(output, d_mat_inv_sqrt)
    return output


def rbf_kernel(input, gamma=-1 / 8.):
    dist = tf.reduce_sum(tf.square(input), -1, keepdims=True)
    dist_trans = tf.transpose(dist, perm=[0, 2, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(input, tf.transpose(input, perm=[0, 2, 1])))),
                      dist_trans)
    kernel = tf.exp(tf.multiply(gamma, sq_dists))

    return kernel
