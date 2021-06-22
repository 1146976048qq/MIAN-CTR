# -*- coding: utf-8 -*-

"""
 * @file    graph_layers.py
 * @author  chenye@antfin.com
 * @date    2019/7/21 4:31 PM
 * @brief   
"""

import tensorflow as tf


class GHMC_Loss:
    def __init__(self, bins=10, momentum=0., bias_correction=True):
        self.bins = bins
        self.momentum = momentum
        self.acc_sum = tf.zeros((bins,))
        self.step = 1
        self.bias_correction = bias_correction

    def calc(self, logits=None, labels=None):
        """ Args:
        logits [batch_num, class_num]:
            The direct prediction of classification fc layer.
        labels [batch_num, class_num]:
            Binary labels (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """

        # gradient length
        g = tf.abs(tf.sigmoid(logits) - labels)

        bin_indices = tf.histogram_fixed_width_bins(g, [0.0, 1.0], nbins=self.bins)
        bin_count = tf.bincount(bin_indices, minlength=self.bins, maxlength=self.bins,
                                dtype=tf.float32)
        n = tf.reduce_sum(tf.cast(bin_count > 0, tf.float32))  # n valid bins

        weights = tf.ones_like(logits) / n

        self.acc_sum = self.momentum * self.acc_sum + (1 - self.momentum) * bin_count
        if self.bias_correction:
            self.acc_sum = self.acc_sum / (1 - tf.pow(self.momentum, self.step))

        for i in range(self.bins):
            condition = tf.equal(bin_indices, i)
            weights = tf.where(condition, weights / self.acc_sum[i], weights)

        weights = tf.stop_gradient(weights)

        loss = tf.reduce_sum(
            tf.multiply(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels),
                weights)
        )

        return loss
