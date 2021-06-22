# -*- coding: utf-8 -*-

"""
 * @file    graph_layers.py
 * @author  chenye@antfin.com
 * @date    2019/8/11 2:37 PM
 * @brief   
"""

import tensorflow as tf
import logging
import keras

from alps.common.model import BaseModel
from alps.common.layers.keras.layer_parser import LayerParser
from keras.layers import *

from keras.engine import Model
from alps.common.layers.keras.ast_parser import add
from utils import build_addon_branch

class ModelESMM(BaseModel):
    def __init__(self, config):
        super(ModelESMM, self).__init__(config)
        pass

    def get_column(self, name):
        for item in self.config.x:
            if item.feature_name == name:
                return item
        return None

    def build_model(self, inputs, labels):

        embedding_dim = self.get_column("deep_features_sparse").embedding_dim
        print("*****chenye*****0530*****embedding_dim:", embedding_dim)

        if self.get_column('deep_features_dense'):
            self.config.use_dense = True

        if self.get_column('wide_features'):
            self.config.use_wide = False

        logging.info("embedding_dim: %s" % embedding_dim)
        logging.info("use_dense: %s" % self.config.use_dense)

        parser = LayerParser(self.config, inputs)

        # deep_input_emb的shape(group_num, sample_num, embedding_dim)
        deep_input, deep_input_emb = parser.get_layer0(("deep_features_sparse", embedding_dim))

        input_list = []
        add(input_list, deep_input)
        print("*****chenye*****0530*****input_list:", input_list)

        # wide_input_emb的shape（1，sample_num, 1）
        if self.config.use_wide:
            ctr_wide_input, ctr_wide_input_emb = parser.get_layer0(("wide_features", 1))
            cvr_wide_input, cvr_wide_input_emb = parser.get_layer0(("wide_features", 1))
        else:
            with tf.variable_scope('ctr_embed'):
                ctr_wide_input, ctr_wide_input_emb = parser.get_layer0(("deep_features_sparse", 1))
            with tf.variable_scope('cvr_embed'):
                cvr_wide_input, cvr_wide_input_emb = parser.get_layer0(("deep_features_sparse", 1))

        add(input_list, ctr_wide_input)
        add(input_list, cvr_wide_input)

        dnn_emb_ctr = Concatenate(axis=-1)(deep_input_emb)

        addon_branch = self.config.model_def.get('addon_branch', None)
        if addon_branch:
            addon_branch_args = self.config.model_def.get('addon_branch_args', {})
            logging.info("addon branch: %s" % addon_branch)
            addon_embed = build_addon_branch(addon_branch, deep_input_emb, addon_branch_args)
            dnn_emb_ctr = Concatenate(axis=-1)([dnn_emb_ctr, addon_embed])

        # deep dense，如果有才运行

        if self.config.use_dense:
            dense_input, dense_input_tmp = parser.get_layer0("deep_features_dense")
            add(input_list, dense_input)

            act = self.config.model_def.get('activation', 'relu')
            layer_dims = self.config.model_def.get('deep_layers_dim', [128, 32, 8, 1])
            l1 = self.config.model_def.get('l1', 0)
            l2 = self.config.model_def.get('l2', 1e-3)

            # 对dense值处理
            if self.config.model_def.dense_batch_norm:
                momentum = self.config.model_def.get('batch_norm_momentum', 0.99)
                dense_input = BatchNormalization(momentum=momentum)(dense_input)
                dense_input = Dense(dense_input.shape[1].value, init='he_normal',
                                    W_regularizer=regularizers.L1L2(l1, l2),
                                    b_regularizer=regularizers.L1L2(l1, l2),
                                    name='fc_deep_input')(dense_input)

            dnn_emb_ctr = Concatenate(axis=-1)([dnn_emb_ctr, dense_input])



        with tf.variable_scope('ctr'):
            x = dnn_emb_ctr

            for i, layer_dim in enumerate(layer_dims):
                x = Activation(act)(x)

                dense = Dense(layer_dim, init='he_normal',
                              W_regularizer=regularizers.L1L2(l1, l2),
                              b_regularizer=regularizers.L1L2(l1, l2),
                              name='{}_fc{}_{}'.format('ctr', i, layer_dim))
                x = dense(x)

            #ctr_logits = merge([x, ctr_wide_input_emb[0]], mode='sum')
            ctr_logits = keras.layers.add([x, ctr_wide_input_emb[0]])

        with tf.variable_scope('cvr'):
            x = dnn_emb_ctr

            for i, layer_dim in enumerate(layer_dims):
                x = Activation(act)(x)

                dense = Dense(layer_dim, init='he_normal',
                              W_regularizer=regularizers.L1L2(l1, l2),
                              b_regularizer=regularizers.L1L2(l1, l2),
                              name='{}_fc{}_{}'.format('cvr', i, layer_dim))
                x = dense(x)

            # cvr_logits = merge([x, cvr_wide_input_emb[0]], mode='sum')
            cvr_logits = keras.layers.add([x, cvr_wide_input_emb[0]])

        ctcvr_logits = Lambda(lambda x: tf.multiply(x[0], x[1]))([ctr_logits, cvr_logits])

        model = Model(input_list, ctcvr_logits, name="ModelESMM")
        model.summary()

        label_click = tf.cast(labels['label'] >= 1, tf.float32)
        label_used = tf.cast(labels['label'] >= 2, tf.float32)

        # 计算loss，分别计算ctr_loss和cvr_loss，然后加权求和
        ctr_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label_click, logits=ctr_logits, name='ctr_loss'))
        ctcvr_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label_used, logits=ctcvr_logits, name='ctcvr_loss'))

        self.loss = ctr_loss + ctcvr_loss
        self.predict_result = tf.sigmoid(ctcvr_logits)
        pred_click = tf.sigmoid(ctr_logits)

        # 计算准确率v
        accuracy_click = tf.reduce_mean(tf.cast(tf.equal(label_click, tf.round(pred_click)), dtype=tf.float32))

        auc_click, auc_op_click = tf.metrics.auc(
            labels=label_click,
            predictions=pred_click, num_thresholds=10240)

        ctr_click = tf.reduce_sum(label_click) / tf.cast(tf.shape(label_click)[0], tf.float32)
        pcopc_click = tf.reduce_sum(pred_click) / tf.reduce_sum(label_click)
        pos_pred_avg_click = tf.div(tf.reduce_sum(pred_click * label_click),
                                    tf.maximum(tf.reduce_sum(label_click), 1))
        neg_pred_avg_click = tf.div(tf.reduce_sum(pred_click * (1 - label_click)),
                                    tf.maximum(tf.reduce_sum(1 - label_click), 1))

        auc_used, auc_op_used = tf.metrics.auc(
            labels=label_used,
            predictions=self.predict_result, num_thresholds=10240)

        # 计算准确率v
        accuracy_used = tf.reduce_mean(tf.cast(tf.equal(label_used, tf.round(self.predict_result)), dtype=tf.float32))

        cvr_used = tf.reduce_sum(label_used) / tf.cast(tf.shape(label_used)[0], tf.float32)
        pcopc_used = tf.reduce_sum(self.predict_result) / tf.reduce_sum(label_used)
        pos_pred_avg_used = tf.div(tf.reduce_sum(self.predict_result * label_used),
                                   tf.maximum(tf.reduce_sum(label_used), 1))
        neg_pred_avg_used = tf.div(tf.reduce_sum(self.predict_result * (1 - label_used)),
                                   tf.maximum(tf.reduce_sum(1 - label_used), 1))

        self.metrics = {'loss': self.loss,
                        'ctr_loss': ctr_loss,
                        'accuracy_used': accuracy_used,
                        'auc_used': auc_op_used,
                        'cvr_used': cvr_used,
                        'pcopc_used': pcopc_used,
                        'accuracy_click': accuracy_click,
                        'auc_click': auc_op_click,
                        'ctr_click': ctr_click,
                        'pcopc_click': pcopc_click,
                        'label_click_sum': tf.reduce_sum(label_click),
                        'label_used_sum': tf.reduce_sum(label_used)
                        }

        tf.summary.scalar("01.train_cvr_loss", self.loss)
        tf.summary.scalar("02.train_ctr_loss", ctr_loss)

        tf.summary.scalar("11.train_auc_used", auc_used)
        tf.summary.scalar("12.cvr_used", cvr_used)
        tf.summary.scalar("13.pcopc_used", pcopc_used)
        tf.summary.scalar("14.pos_pred_avg_used", pos_pred_avg_used)
        tf.summary.scalar("15.neg_pred_avg_used", neg_pred_avg_used)

        tf.summary.scalar("21.train_auc_click", auc_click)
        tf.summary.scalar("22.ctr_click", ctr_click)
        tf.summary.scalar("23.pcopc_click", pcopc_click)
        tf.summary.scalar("24.pos_pred_avg_click", pos_pred_avg_click)
        tf.summary.scalar("25.neg_pred_avg_click", neg_pred_avg_click)

        # calibration
        # self.predict_result = self.predict_result / (self.predict_result + (1 - self.predict_result) / 0.06)

        return self.predict_result

    def get_prediction_result(self, **options):
        return None

    def get_loss(self, **options):
        return self.loss

    def get_metrics(self, **kwargs):
        return self.metrics

    def get_summary_op(self):
        return tf.summary.merge_all(), None

    @property
    def name(self):
        return 'ModelESMM'
