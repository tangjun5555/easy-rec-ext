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

    def _build(self, mode):
        file_paths = []
        for x in self._input_path.split(","):
            file_paths.extend(tf.gfile.Glob(x))
        assert len(file_paths) > 0, "match no files with %s" % self._input_path

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging.info("train files[%d]: %s" % (len(file_paths), ",".join(file_paths)))

            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            dataset = dataset.shuffle(len(file_paths))
            # too many readers read the same file will cause performance issues
            # as the same data will be read multiple times
            parallel_num = min(self._num_parallel_calls, len(file_paths))
            dataset = dataset.interleave(
                tf.data.TextLineDataset, cycle_length=parallel_num, num_parallel_calls=parallel_num
            )

            dataset = dataset.shuffle(self._shuffle_buffer_size, seed=555, reshuffle_each_iteration=True)
            dataset = dataset.repeat(self._input_config.num_epochs)
        else:
            logging.info("eval files[%d]: %s" % (len(file_paths), ",".join(file_paths)))
            dataset = tf.data.TextLineDataset(file_paths)
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self._input_config.batch_size)
        dataset = dataset.map(self._parse_csv, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.prefetch(buffer_size=self._prefetch_size)
        dataset = dataset.map(map_func=self._preprocess, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.prefetch(buffer_size=self._prefetch_size)

        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.map(lambda x:
                                  (self._get_features(x), self._get_labels(x)))
        else:
            dataset = dataset.map(lambda x: (self._get_features(x)))
        return dataset
