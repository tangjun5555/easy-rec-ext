# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/9/12 5:43 下午
# desc:

import os
import json
import logging

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
    SessionRunHook = tf.estimator.SessionRunHook
    CheckpointSaverHook = tf.estimator.CheckpointSaverHook
else:
    SessionRunHook = tf.train.SessionRunHook
    CheckpointSaverHook = tf.train.CheckpointSaverHook


def parse_tf_config():
    tf_config_str = os.environ.get("TF_CONFIG", "")
    if "TF_CONFIG" in os.environ:
        tf_config = json.loads(tf_config_str)
        # {"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}
        cluster = tf_config["cluster"]
        task = tf_config["task"]
        task_type = task["type"]
        task_index = task["index"]
    else:
        cluster = {}
        task_type = "master"
        task_index = 0
    logging.info("%s parse_tf_config, cluster:%s, task_type:%s, task_index:%s" %
                 (str(os.path.basename(__file__)).split(".")[0], str(cluster), str(task_type), str(task_index))
                 )
    return cluster, task_type, task_index


def get_task_index_and_num():
    cluster, task_type, task_index = parse_tf_config()
    if "worker" not in cluster:
        return 0, 1
    if task_type == "evaluator":
        return 0, 1
    task_num = len(cluster["worker"])
    if "chief" in cluster or "master" in cluster:
        task_num += 1
        if task_type not in ["chief", "master"]:
            task_index += 1
    logging.info("%s get_task_index_and_num, task_index:%s, task_num:%s" %
                 (str(os.path.basename(__file__)).split(".")[0], str(task_index), str(task_num))
                 )
    return task_index, task_num


def chief_to_master():
    if "TF_CONFIG" in os.environ:
        tf_config = json.loads(os.environ["TF_CONFIG"])
        # change chief to master
        if "chief" in tf_config["cluster"]:
            tf_config["cluster"]["master"] = tf_config["cluster"]["chief"]
            del tf_config["cluster"]["chief"]
            if tf_config["task"]["type"] == "chief":
                tf_config["task"]["type"] = "master"
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        return tf_config
    else:
        return None
