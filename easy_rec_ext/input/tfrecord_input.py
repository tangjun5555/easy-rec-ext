# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 4:57 下午
# desc:

import logging
from easy_rec_ext.input.input import Input

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class TFRecordInput(Input):
    def __init__(self,
                 input_config,
                 feature_config,
                 input_path: str,
                 task_index=0,
                 task_num=1,
                 ):
        super(TFRecordInput, self).__init__(input_config, feature_config, input_path,
                                            task_index, task_num)

    def _build(self, mode):
        file_paths = []
        for x in self._input_path.split(","):
            file_paths.extend(tf.gfile.Glob(x))
        file_paths = sorted(file_paths)
        assert len(file_paths) > 0, "match no files with %s" % self._input_path

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging.info("train files[%d]: %s" % (len(file_paths), ",".join(file_paths)))

            dataset = tf.data.Dataset.from_tensor_slices(file_paths)
            # dataset = dataset.shuffle(len(file_paths))

            # too many readers read the same file will cause performance issues
            # as the same data will be read multiple times
            parallel_num = min(self._num_parallel_calls, len(file_paths))
            dataset = dataset.interleave(
                tf.data.TFRecordDataset, cycle_length=parallel_num, num_parallel_calls=parallel_num
            )
            dataset = dataset.shard(self._task_num, self._task_index)

            dataset = dataset.shuffle(buffer_size=self._shuffle_buffer_size, seed=555, reshuffle_each_iteration=True)
            dataset = dataset.repeat(self._input_config.num_epochs)
        else:
            logging.info("eval files[%d]: %s" % (len(file_paths), ",".join(file_paths)))
            dataset = tf.data.TFRecordDataset(file_paths)
            dataset = dataset.repeat(1)

        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.batch(self._input_config.batch_size)
        dataset = dataset.prefetch(buffer_size=self._prefetch_size)
        dataset = dataset.map(map_func=self._preprocess, num_parallel_calls=self._num_parallel_calls)
        dataset = dataset.prefetch(buffer_size=self._prefetch_size)

        if mode != tf.estimator.ModeKeys.PREDICT:
            dataset = dataset.map(lambda x:
                                  (self._get_features(x), self._get_labels(x)))
        else:
            dataset = dataset.map(lambda x: (self._get_features(x)))
        return dataset
