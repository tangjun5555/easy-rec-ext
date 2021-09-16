# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/27 12:24 下午
# desc:

import os
from abc import abstractmethod
from collections import OrderedDict

from easy_rec_ext.utils import string_ops
from easy_rec_ext.core.pipeline import InputConfig, FeatureConfig

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class Input(object):
    def __init__(self,
                 input_config: InputConfig,
                 feature_config: FeatureConfig,
                 input_path: str,
                 task_index=0,
                 task_num=1,
                 ):
        self._input_config = input_config
        self._feature_config = feature_config
        self._input_path = input_path
        self._task_index = task_index
        self._task_num = task_num

        self._input_fields = [x.input_name for x in self._input_config.input_fields]
        self._input_field_types = [x.input_type for x in self._input_config.input_fields]

        self._effective_fields = []
        for feature_field in self._feature_config.feature_fields:
            if feature_field.input_name not in self._effective_fields:
                self._effective_fields.append(feature_field.input_name)
        self._effective_fids = [self._input_fields.index(x) for x in self._effective_fields]

        self._label_fields = list(self._input_config.label_fields)
        self._label_fids = [self._input_fields.index(x) for x in self._label_fields]

        self._input_field_defaults = [self.get_type_defaults(t) for t in self._input_field_types]

        self._num_parallel_calls = 8
        self._prefetch_size = 32
        self._shuffle_buffer_size = 32

    def get_tf_type(self, field_type):
        type_map = {
            "int": tf.dtypes.int64,
            "string": tf.dtypes.string,
            "float": tf.dtypes.float32,
        }
        assert field_type in type_map, "invalid type: %s" % field_type
        return type_map[field_type]

    def get_type_defaults(self, field_type):
        type_defaults = {
            "int": 0,
            "string": "",
            "float": 0.0,
        }
        assert field_type in type_defaults, "invalid type: %s" % field_type
        return type_defaults[field_type]

    def _parse_csv(self, line):
        def _check_data(line):
            sep = ","
            if type(sep) != type(str):
                sep = sep.encode("utf-8")
            field_num = len(line[0].split(sep))
            assert field_num == len(self._input_field_defaults), \
                "sep[%s] maybe invalid: field_num=%d, required_num=%d" % (
                sep, field_num, len(self._input_field_defaults))
            return True

        check_op = tf.py_func(_check_data, [line], Tout=tf.bool)
        with tf.control_dependencies([check_op]):
            fields = tf.decode_csv(line, record_defaults=self._input_field_defaults, field_delim=",", name="decode_csv")

        inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
        for x in self._label_fids:
            inputs[self._input_fields[x]] = fields[x]
        return inputs

    def _get_features(self, fields):
        field_dict = {x: fields[x] for x in self._effective_fields if x in fields}
        return field_dict

    def _get_labels(self, fields):
        return OrderedDict([
            (x,
             tf.squeeze(fields[x], axis=1)
             if len(fields[x].get_shape()) == 2 and fields[x].get_shape()[1] == 1
             else fields[x]
             )
            for x in self._label_fields
        ])

    @abstractmethod
    def _build(self, mode):
        pass

    def _preprocess(self, field_dict):
        """Preprocess the feature columns.
    
        preprocess some feature columns, such as TagFeature or LookupFeature,
        it is expected to handle batch inputs and single input,
        it could be customized in subclasses
    
        Args:
          field_dict: string to tensor, tensors are dense,
              could be of shape [batch_size], [batch_size, None], or of shape []
    
        Returns:
          output_dict: some of the tensors are transformed into sparse tensors,
              such as input tensors of tag features and lookup features
        """
        parsed_dict = {}
        for fc in self._feature_config.feature_fields:
            if fc.feature_type == "SequenceFeature":
                field = field_dict[fc.input_name]
                parsed_dict[fc.input_name] = tf.strings.split(field, "|")
                if fc.hash_bucket_size <= 0:
                    parsed_dict[fc.input_name] = tf.sparse.SparseTensor(
                        parsed_dict[fc.input_name].indices,
                        tf.string_to_number(parsed_dict[fc.input_name].values, tf.int64,
                                            name="sequence_str_2_int_%s" % fc.input_name),
                        parsed_dict[fc.input_name].dense_shape
                    )
                else:
                    parsed_dict[fc.input_name] = tf.sparse.SparseTensor(
                        parsed_dict[fc.input_name].indices,
                        string_ops.string_to_hash_bucket(parsed_dict[fc.input_name].values, fc.hash_bucket_size),
                        parsed_dict[fc.input_name].dense_shape
                    )
                parsed_dict[fc.input_name] = tf.sparse.to_dense(parsed_dict[fc.input_name])

            elif fc.feature_type == "RawFeature":
                if field_dict[fc.input_name].dtype == tf.string:
                    if fc.raw_input_dim > 1:
                        tmp_fea = tf.string_split(field_dict[fc.input_name], "|")
                        tmp_vals = tf.string_to_number(tmp_fea.values, tf.float32,
                                                       name="multi_raw_fea_to_flt_%s" % fc.input_name)
                        parsed_dict[fc.input_name] = tf.sparse_to_dense(
                            tmp_fea.indices,
                            [tf.shape(field_dict[fc.input_name])[0], fc.raw_input_dim],
                            tmp_vals,
                            default_value=0,
                        )
                    else:
                        parsed_dict[fc.input_name] = tf.string_to_number(field_dict[fc.input_name], tf.float32)
                        parsed_dict[fc.input_name] = tf.expand_dims(parsed_dict[fc.input_name], axis=1)
                else:
                    parsed_dict[fc.input_name] = tf.to_float(field_dict[fc.input_name])
                    parsed_dict[fc.input_name] = tf.expand_dims(parsed_dict[fc.input_name], axis=1)

            elif fc.feature_type == "IdFeature":
                parsed_dict[fc.input_name] = field_dict[fc.input_name]
                if fc.hash_bucket_size <= 0:
                    if parsed_dict[fc.input_name].dtype == tf.string:
                        parsed_dict[fc.input_name] = tf.string_to_number(parsed_dict[fc.input_name], tf.dtypes.int64,
                                                                         name="%s_str_2_int" % fc.input_name)
                else:
                    if "use_dynamic_embedding" in os.environ and os.environ["use_dynamic_embedding"] == "1":
                        pass
                    else:
                        parsed_dict[fc.input_name] = string_ops.string_to_hash_bucket(
                            parsed_dict[fc.input_name],
                            fc.hash_bucket_size
                        )
                parsed_dict[fc.input_name] = tf.expand_dims(parsed_dict[fc.input_name], axis=1)
            else:
                parsed_dict[fc.input_name] = field_dict[fc.input_name]

        for input_id, input_name in enumerate(self._label_fields):
            if input_name not in field_dict:
                continue
            assert field_dict[input_name].dtype \
                   in [tf.float32, tf.double, tf.int32, tf.int64], "invalid label dtype: %s" % str(
                field_dict[input_name].dtype)
            parsed_dict[input_name] = field_dict[input_name]

        return parsed_dict

    def create_placeholders(self):
        self._mode = tf.estimator.ModeKeys.PREDICT
        inputs_placeholder = tf.placeholder(tf.string, [None], name="features")
        input_vals = tf.string_split(inputs_placeholder, ",", skip_empty=False).values
        effective_fids = list(self._effective_fids)
        input_vals = tf.reshape(input_vals, [-1, len(effective_fids)], name="input_reshape")
        features = {}
        for tmp_id, fid in enumerate(effective_fids):
            ftype = self._input_field_types[fid]
            tf_type = self.get_tf_type(ftype)
            input_name = self._input_fields[fid]
            if tf_type in [tf.float32, tf.double, tf.int32, tf.int64]:
                features[input_name] = tf.string_to_number(input_vals[:, tmp_id], tf_type,
                                                           name="input_str_to_%s" % tf_type.name)
            else:
                features[input_name] = input_vals[:, tmp_id]
        features = self._preprocess(features)
        return {"features": inputs_placeholder}, features

    def create_input(self, export_config=None):
        def _input_fn(mode=None, params=None, config=None):
            """Build input_fn for estimator.
      
            Args:
              mode: tf.estimator.ModeKeys.(TRAIN, EVAL, PREDICT)
                  params: `dict` of hyper parameters, from Estimator
                  config: tf.estimator.RunConfig instance
      
            Return:
              if mode is not None, return:
                  features: inputs to the model.
                  labels: groundtruth
              else, return:
                  tf.estimator.export.ServingInputReceiver instance
            """
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
                self._mode = mode
                dataset = self._build(mode)
                return dataset
            else:
                # serving_input_receiver_fn for export SavedModel
                inputs, features = self.create_placeholders()
                return tf.estimator.export.ServingInputReceiver(features, inputs)

        return _input_fn
