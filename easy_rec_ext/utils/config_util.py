# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/11/3 6:28 下午
# desc:

import os
import json

import logging

from tensorflow.python.lib.io import file_io

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def save_pipeline_config(pipeline_config, directory, filename="pipeline.config"):
    """Saves a pipeline config text file to disk.
  
    Args:
      pipeline_config:
      directory: The model directory into which the pipeline config file will be saved.
      filename: pipelineconfig filename
    """
    if not file_io.file_exists(directory):
        file_io.recursive_create_dir(directory)

    pipeline_config_path = os.path.join(directory, filename)
    with tf.gfile.Open(pipeline_config_path, "wb") as f:
        logging.info("Writing protobuf message file to %s", filename)
        f.write(json.dumps(pipeline_config))
