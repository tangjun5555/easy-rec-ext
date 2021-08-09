# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/27 12:19 下午
# desc:

import logging
from easy_rec_ext.input.input import Input

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class CSVInput(Input):
    def __init__(self, input_config, feature_config, input_path):
        super(CSVInput, self).__init__(input_config, feature_config, input_path)
        self._record_defaults = [self.get_type_defaults(t) for t in self._input_field_types]

    def _parse_csv(self, line):
        def _check_data(line):
            sep = ","
            if type(sep) != type(str):
                sep = sep.encode("utf-8")
            field_num = len(line[0].split(sep))
            assert field_num == len(self._record_defaults), \
                "sep[%s] maybe invalid: field_num=%d, required_num=%d" % (sep, field_num, len(self._record_defaults))
            return True

        check_op = tf.py_func(_check_data, [line], Tout=tf.bool)
        with tf.control_dependencies([check_op]):
            fields = tf.decode_csv(line, record_defaults=self._record_defaults, field_delim=",", name="decode_csv")

        inputs = {self._input_fields[x]: fields[x] for x in self._effective_fids}
        for x in self._label_fids:
            inputs[self._input_fields[x]] = fields[x]
        return inputs

    def _build(self, mode):
        file_paths = []
        for x in self._input_path.split(","):
            file_paths.extend(tf.gfile.Glob(x))
        assert len(file_paths) > 0, "match no files with %s" % self._input_path

        num_parallel_calls = 8
        if mode == tf.estimator.ModeKeys.TRAIN:
            logging.info("train files[%d]: %s" % (len(file_paths), ",".join(file_paths)))
            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            dataset = dataset.shuffle(len(file_paths))
            # too many readers read the same file will cause performance issues
            # as the same data will be read multiple times
            parallel_num = min(num_parallel_calls, len(file_paths))
            dataset = dataset.interleave(tf.data.TextLineDataset, cycle_length=parallel_num,
                                         num_parallel_calls=parallel_num)
            dataset = dataset.shuffle(32, seed=555, reshuffle_each_iteration=True)
            dataset = dataset.repeat(self._input_config.num_epochs)
        else:
            logging.info("eval files[%d]: %s" % (len(file_paths), ",".join(file_paths)))
            dataset = tf.data.TextLineDataset(file_paths)
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self._input_config.batch_size)
        dataset = dataset.map(self._parse_csv, num_parallel_calls=num_parallel_calls)
        dataset = dataset.prefetch(buffer_size=32)
        dataset = dataset.map(map_func=self._preprocess, num_parallel_calls=num_parallel_calls)
        dataset = dataset.prefetch(buffer_size=32)

        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.map(lambda x:
                                  (self._get_features(x), self._get_labels(x)))
        else:
            dataset = dataset.map(lambda x: (self._get_features(x)))
        return dataset
