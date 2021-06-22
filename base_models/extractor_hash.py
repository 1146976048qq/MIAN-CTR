# -*- coding: utf-8 -*-

"""
 * @file    graph_layers.py
 * @author  chenye@antfin.com
 * @date    2019/8/12 2:07 PM
 * @brief   
"""

# 说明: 为了适应去feature map，重写了feature extractor

from alps.common.processor import BaseFeatureExtractor
from alps.common.context import *
from alps.common.utils.type_util import get_type_convert
import numpy as np
from scipy.sparse import csr_matrix


class ExtractorHash(BaseFeatureExtractor):

    def extract_sparse(
            self,
            batch_record,
            shape,
            converter=None,
            dtype=None,
            separator='\001',
            kv_separator=None,
            group_separator='\002',
            group=None,
            column=None,
            feature_id_map=None,
            id_as_value=True):
        """
        convert batch record into sparse feature as tuple of numpy array

        Args:
          batch_record:
          shape:
          converter:
          dtype:
          separator:
          kv_separator:
          group_separator:
          group:
          column:
          feature_id_map:
          id_as_value:

        Returns:
          a list of numpy array [(indices, values, shapes), (indices, values, shapes)]

        """

        if len(batch_record) > 0 and isinstance(batch_record[0], csr_matrix):
            result_indices = []
            result_values = []
            result_shape = [len(batch_record), shape]
            for idx, record in enumerate(batch_record):
                sparse_coo = record.tocoo()
                indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                          np.expand_dims(sparse_coo.col, 1)), 1)
                indices = [(idx, indice[1]) for indice in indices]
                values = np.array(sparse_coo.data, dtype=np.float32)
                result_indices.extend(indices)
                result_values.extend(values)

            return [tuple([result_indices, result_values, result_shape])]

        if dtype is None:
            dtype = 'float'

        if converter is None:
            converter = get_type_convert(dtype)

        if group is None:
            group = 1

        _shape = [len(batch_record), shape]

        indices = [[] for _ in xrange(group)]
        values = [[] for _ in xrange(group)]
        shapes = [_shape for _ in xrange(group)]

        unseen_groups = set(xrange(group))

        for idx, record in enumerate(batch_record):
            if record is None or len(record) == 0:
                continue
            if isinstance(record, (str, unicode)):
                record = record.split(separator)
            elif not isinstance(record, list):
                record = [record]

            _indices_dic = [[] for _ in xrange(group)]
            _values_dic = [[] for _ in xrange(group)]

            for item in record:
                if len(item) == 0:
                    continue
                if group > 1:
                    kv, group_id = item.split(group_separator)
                    group_id = int(group_id)
                else:
                    kv = item
                    group_id = 0

                if kv_separator is None:
                    id = int(kv)
                    id %= shape
                    value = id
                else:
                    id, value = kv.split(kv_separator)
                    id = int(id)
                    id %= shape
                    value = id

                assert group_id < group, 'group id %s must small than group num %s' % (group_id, group)

                _indices_dic[group_id].append([idx, int(id)])
                _values_dic[group_id].append(value)

                if group_id in unseen_groups:
                    unseen_groups.remove(group_id)

            for group_id, _indices in zip(range(group), _indices_dic):
                _value = _values_dic[group_id]
                indices[group_id].extend(_indices)
                values[group_id].extend(_value)

        result = []
        group_id = 0
        for _indices, _values, _shapes in zip(indices, values, shapes):
            _indices = np.array(_indices, dtype=np.int32)
            if group_id in unseen_groups:
                _indices = np.concatenate((np.expand_dims(_indices, 1), np.expand_dims(_indices, 1)), 1)
            _values = np.array(_values, dtype=dtype)
            _shapes = np.array(_shapes, dtype=np.int32)
            group_value = (_indices, _values, _shapes)
            result.append(group_value)
            group_id += 1

        return result
