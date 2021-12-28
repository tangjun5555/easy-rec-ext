# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:07 下午
# desc:

import logging
import json
import os
import argparse

from easy_rec_ext.core.pipeline import PipelineConfig
from easy_rec_ext.core.exporter import FinalExporter
from easy_rec_ext.builders import distribute_strategy_builder
from easy_rec_ext.input import CSVInput, TFRecordInput, OSSInput
from easy_rec_ext.model.rank_estimator import RankEstimator
from easy_rec_ext.utils import estimator_util, config_util

import tensorflow as tf

version = "0.1.2"

if tf.__version__ >= "2.0":
    gfile = tf.compat.v1.gfile

    from tensorflow.core.protobuf import config_pb2

    GPUOptions = config_pb2.GPUOptions
    ConfigProto = config_pb2.ConfigProto
else:
    gfile = tf.gfile

    GPUOptions = tf.GPUOptions
    ConfigProto = tf.ConfigProto

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_config_path", type=str, required=True)
parser.add_argument("--task_type", type=str, required=True,
                    choices=["train_and_evaluate", "evaluate", "predict", "export"])
parser.add_argument("--model_dir", type=str, required=False)
parser.add_argument("--train_input_path", type=str, required=False)
parser.add_argument("--eval_input_path", type=str, required=False)
parser.add_argument("--export_dir", type=str, required=False)
parser.add_argument("--log_level", type=str, required=False, default="info")
args = parser.parse_args()
print("Run params:" + str(args))

if args.log_level == "dubug":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %a",
    )
elif args.log_level == "warn":
    logging.basicConfig(
        level=logging.WARN,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %a",
    )
elif args.log_level == "error":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %a",
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %a",
    )


def _check_model_dir(model_dir, continue_train):
    if not continue_train:
        if not gfile.IsDirectory(model_dir):
            gfile.MakeDirs(model_dir)
        else:
            assert len(gfile.Glob(model_dir + "/model.ckpt-*.meta")) == 0, \
                "model_dir[=%s] already exists and not empty(if you " \
                "want to continue train on current model_dir please " \
                "delete dir %s or specify --continue_train[internal use only])" % (
                    model_dir, model_dir)
    else:
        if not gfile.IsDirectory(model_dir):
            logging.info("%s does not exists, create it automatically" % model_dir)
            gfile.MakeDirs(model_dir)


def get_pipeline_config_from_file(pipeline_config_path) -> PipelineConfig:
    with open(pipeline_config_path, "r") as f:
        res = json.load(f)
        res = PipelineConfig.handle(res)
    if args.model_dir:
        logging.info("use args, model_dir:" + str(args.model_dir))
        res.model_dir = args.model_dir
    if args.train_input_path:
        logging.info("use args, train_input_path:" + str(args.train_input_path))
        res.input_config.train_input_path = args.train_input_path
    if args.eval_input_path:
        logging.info("use args, eval_input_path:" + str(args.eval_input_path))
        res.input_config.eval_input_path = args.eval_input_path
    if args.export_dir:
        logging.info("use args, export_dir:" + str(args.export_dir))
        res.export_config.export_dir = args.export_dir
    logging.info("main get_pipeline_config_from_file, pipeline_config:" + str(res))

    if res.model_config.use_dynamic_embedding:
        os.environ["use_dynamic_embedding"] = "1"
    else:
        os.environ["use_dynamic_embedding"] = "0"
    if "CUDA_VISIBLE_DEVICES" in os.environ \
        and os.environ["CUDA_VISIBLE_DEVICES"] \
        and os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
        os.environ["use_gpu"] = "1"
    else:
        os.environ["use_gpu"] = "0"
    return res


def get_input_fn(input_config, feature_config, input_path=None, export_config=None):
    """
    Build estimator input function.

    Args:
      input_config: dataset config
      feature_config: FeatureConfig
      input_path: input_data_path
      export_config: configuration for exporting models, only used to build input_fn when exporting models

    Returns:
      subclass of Input
    """
    task_id, task_num = estimator_util.get_task_index_and_num()
    if input_config.input_type == "csv":
        input_obj = CSVInput(input_config, feature_config, input_path, task_index=task_id, task_num=task_num)
    elif input_config.input_type == "tfrecord":
        input_obj = TFRecordInput(input_config, feature_config, input_path, task_index=task_id, task_num=task_num)
    elif input_config.input_type == "oss":
        input_obj = OSSInput(input_config, feature_config, input_path, task_index=task_id, task_num=task_num)
    else:
        raise ValueError("input_type:%s not supported." % input_config.input_type)
    input_fn = input_obj.create_input(export_config)
    return input_fn


def _get_ckpt_path(pipeline_config):
    ckpt_path = tf.train.latest_checkpoint(pipeline_config.model_dir)
    logging.info("use latest checkpoint %s from %s" % (ckpt_path, pipeline_config.model_dir))
    return ckpt_path


def _create_estimator(pipeline_config, distribution=None):
    session_config = ConfigProto(
        gpu_options=GPUOptions(allow_growth=False),
        allow_soft_placement=True,
        log_device_placement=True,
        # inter_op_parallelism_threads=pipeline_config.train_config.inter_op_parallelism_threads,
        # intra_op_parallelism_threads=pipeline_config.train_config.intra_op_parallelism_threads
    )
    session_config.device_filters.append("/job:ps")

    run_config = tf.estimator.RunConfig(
        model_dir=pipeline_config.model_dir,
        log_step_count_steps=pipeline_config.train_config.log_step_count_steps,
        save_checkpoints_steps=pipeline_config.train_config.save_checkpoints_steps,
        keep_checkpoint_max=pipeline_config.train_config.keep_checkpoint_max,
        train_distribute=distribution,
        eval_distribute=distribution,
        session_config=session_config,
    )
    estimator = RankEstimator(pipeline_config, run_config)
    return estimator, run_config


def _create_eval_export_spec(pipeline_config, eval_data):
    export_input_fn = get_input_fn(pipeline_config.input_config,
                                   pipeline_config.feature_config,
                                   None,
                                   pipeline_config.export_config
                                   )
    exporters = [
        FinalExporter(name="final", serving_input_receiver_fn=export_input_fn)
    ]
    eval_input_fn = get_input_fn(pipeline_config.input_config, pipeline_config.feature_config, eval_data)
    eval_spec = tf.estimator.EvalSpec(
        name="val",
        input_fn=eval_input_fn,
        steps=None,
        throttle_secs=10,
        exporters=exporters
    )
    return eval_spec


def train_and_evaluate(pipeline_config: PipelineConfig):
    """
    
    Args:
        pipeline_config:

    Returns:

    """

    if "TF_CONFIG" in os.environ:
        tf_config = json.loads(os.environ["TF_CONFIG"])

        if "cluster" in tf_config and "chief" in tf_config["cluster"] \
            and "ps" in tf_config["cluster"] \
            and ("evaluator" not in tf_config["cluster"]):
            # chief = tf_config["cluster"]["chief"]
            # del tf_config["cluster"]["chief"]
            # tf_config["cluster"]["master"] = chief
            # if tf_config["task"]["type"] == "chief":
            #     tf_config["task"]["type"] = "master"
            # os.environ["TF_CONFIG"] = json.dumps(tf_config)
            estimator_util.chief_to_master()

    if not gfile.Exists(pipeline_config.model_dir) \
        and pipeline_config.model_config.pretrain_variable_dir \
        and gfile.Exists(pipeline_config.model_config.pretrain_variable_dir):
        os.environ["pretrain_variable_dir"] = pipeline_config.model_config.pretrain_variable_dir

    distribution = distribute_strategy_builder.build(pipeline_config.train_config)

    estimator, _ = _create_estimator(pipeline_config, distribution)

    # master_stat_file = os.path.join(pipeline_config.model_dir, "master.stat")
    version_file = os.path.join(pipeline_config.model_dir, "version")
    if estimator_util.is_chief():
        _check_model_dir(pipeline_config.model_dir, True)
        config_util.save_pipeline_config(pipeline_config, pipeline_config.model_dir)
        with gfile.GFile(version_file, "w") as f:
            f.write(version + "\n")
        # if gfile.Exists(master_stat_file):
        #     gfile.Remove(master_stat_file)

    train_input_fn = get_input_fn(pipeline_config.input_config,
                                  pipeline_config.feature_config,
                                  pipeline_config.input_config.train_input_path
                                  )
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = _create_eval_export_spec(pipeline_config, pipeline_config.input_config.eval_input_path)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    logging.info("Train and evaluate finish")


def evaluate(pipeline_config, eval_result_filename="eval_result.txt"):
    """

    Args:
        pipeline_config:
        eval_result_filename:

    Returns:

    """
    cluster = None
    server_target = None
    if "TF_CONFIG" in os.environ:
        tf_config = estimator_util.chief_to_master()
        from tensorflow.python.training import server_lib
        if tf_config["task"]["type"] == "ps":
            cluster = tf.train.ClusterSpec(tf_config["cluster"])
            server = server_lib.Server(
                cluster, job_name="ps", task_index=tf_config["task"]["index"]
            )
            server.join()
        elif tf_config["task"]["type"] == "master":
            if "ps" in tf_config["cluster"]:
                cluster = tf.train.ClusterSpec(tf_config["cluster"])
                server = server_lib.Server(cluster, job_name="master", task_index=0)
                server_target = server.target
                print("server_target = %s" % server_target)

    distribution = distribute_strategy_builder.build(pipeline_config.train_config)

    estimator, run_config = _create_estimator(pipeline_config, distribution)
    eval_spec = _create_eval_export_spec(pipeline_config, pipeline_config.input_config.eval_input_path)

    ckpt_path = _get_ckpt_path(pipeline_config)

    if server_target:
        # evaluate with parameter server
        input_iter = eval_spec.input_fn(
            mode=tf.estimator.ModeKeys.EVAL).make_one_shot_iterator()
        input_feas, input_lbls = input_iter.get_next()
        from tensorflow.python.training.device_setter import replica_device_setter
        from tensorflow.python.framework.ops import device
        from tensorflow.python.training.monitored_session import MonitoredSession
        from tensorflow.python.training.monitored_session import ChiefSessionCreator

        with device(replica_device_setter(worker_device="/job:master/task:0", cluster=cluster)):
            estimator_spec = estimator._eval_model_fn(input_feas, input_lbls, run_config)

        session_config = ConfigProto(
            allow_soft_placement=True, log_device_placement=True)
        chief_sess_creator = ChiefSessionCreator(
            master=server_target,
            checkpoint_filename_with_path=ckpt_path,
            config=session_config)
        eval_metric_ops = estimator_spec.eval_metric_ops
        update_ops = [eval_metric_ops[x][1] for x in eval_metric_ops.keys()]
        metric_ops = {x: eval_metric_ops[x][0] for x in eval_metric_ops.keys()}
        update_op = tf.group(update_ops)
        with MonitoredSession(session_creator=chief_sess_creator, hooks=None, stop_grace_period_secs=120) as sess:
            while True:
                try:
                    sess.run(update_op)
                except tf.errors.OutOfRangeError:
                    break
            eval_result = sess.run(metric_ops)
    else:
        eval_result = estimator.evaluate(eval_spec.input_fn, eval_spec.steps, checkpoint_path=ckpt_path)
    logging.info("Evaluate finish")

    # write eval result to file
    eval_result_file = os.path.join(pipeline_config.model_dir, eval_result_filename)
    logging.info("save eval result to file %s" % eval_result_file)
    with gfile.GFile(eval_result_file, "w") as ofile:
        result_to_write = {}
        for key in sorted(eval_result):
            # # skip logging binary data
            # if isinstance(eval_result[key], six.binary_type):
            #     continue
            # convert numpy float to python float
            result_to_write[key] = eval_result[key].item()
        ofile.write(json.dumps(result_to_write))
    return eval_result


def predict(pipeline_config):
    """
    Predict a EasyRec model defined in pipeline_config_path.
    Predict the model defined in pipeline_config_path on the eval data.

    Args:
      pipeline_config:

    Returns:
      A list of dict of predict results

    Raises:
      AssertionError, if:
        * pipeline_config_path does not exist
    """
    estimator, _ = _create_estimator(pipeline_config)
    eval_spec = _create_eval_export_spec(pipeline_config, pipeline_config.input_config.eval_input_path)
    ckpt_path = _get_ckpt_path(pipeline_config)
    pred_result = estimator.predict(eval_spec.input_fn, checkpoint_path=ckpt_path)
    logging.info("Predict finish")
    return pred_result


def export(pipeline_config):
    if not gfile.Exists(pipeline_config.export_config.export_dir):
        gfile.MakeDirs(pipeline_config.export_config.export_dir)
    estimator, _ = _create_estimator(pipeline_config)
    serving_input_fn = get_input_fn(pipeline_config.input_config,
                                    pipeline_config.feature_config,
                                    None,
                                    pipeline_config.export_config
                                    )
    final_export_dir = estimator.export_savedmodel(
        export_dir_base=pipeline_config.export_config.export_dir,
        serving_input_receiver_fn=serving_input_fn,
        strip_default_attrs=True
    )
    logging.info("model has been exported to %s successfully" % final_export_dir)
    return final_export_dir


if __name__ == "__main__":
    pipeline_config = get_pipeline_config_from_file(args.pipeline_config_path)
    if "train_and_evaluate" == args.task_type:
        train_and_evaluate(pipeline_config)
    elif "evaluate" == args.task_type:
        evaluate(pipeline_config)
    elif "predict" == args.task_type:
        predict(pipeline_config)
    elif "export" == args.task_type:
        export(pipeline_config)
    else:
        raise ValueError("task_type:%s not supported." % args.task_type)
