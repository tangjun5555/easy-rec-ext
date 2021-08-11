# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/6 7:04 下午
# desc:


import oss2
import logging
from easy_rec_ext.input.input import Input
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class OSSInput(Input):
    def __init__(self, input_config, feature_config, input_path):
        super(OSSInput, self).__init__(input_config, feature_config, input_path)

    def _get_oss_bucket(self):
        auth = oss2.Auth(self._input_config.oss_config.access_key_id, self._input_config.oss_config.access_key_secret)
        bucket = oss2.Bucket(auth, self._input_config.oss_config.endpoint, self._input_config.oss_config.bucket_name)
        return bucket

    def _get_oss_stream(self, path):
        bucket = self._get_oss_bucket()
        return bucket.get_object(path)

    def _get_file_list(self, root_path):
        bucket = self._get_oss_bucket()
        res = []
        for obj in oss2.ObjectIterator(bucket, prefix=root_path, delimiter="/"):
            if not obj.is_prefix():
                res.append(obj.key)
        return res

    def _build(self, mode):
        file_paths = []
        for x in self._input_path.split(","):
            file_paths.extend(self._get_file_list(x))

        def generator_fn():
            for path in file_paths:
                object_stream = self._get_oss_stream(path)
                buffer = ""
                while True:
                    tmp = str(object_stream.read(1024), encoding="utf-8")
                    if not tmp:
                        break
                    buffer += tmp
                    if "\n" in buffer:
                        split = buffer.split("\n")
                        buffer = split[-1]
                        for i in range(len(split) - 1):
                            line = split[i]
                            yield line

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging.info("train files[%d]: %s" % (len(file_paths), ",".join(file_paths)))
            dataset = tf.data.Dataset.from_generator(
                generator=generator_fn,
                output_signature=tf.TensorSpec(shape=(), dtype=tf.dtypes.string)
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
