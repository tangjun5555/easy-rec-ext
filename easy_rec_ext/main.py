# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:07 下午
# desc:

import logging
import json
import os
import argparse
import tensorflow as tf
from easy_rec_ext.core.pipeline import PipelineConfig
from easy_rec_ext.core.exporter import FinalExporter
from easy_rec_ext.input import CSVInput, TFRecordInput
from easy_rec_ext.model.rank_estimator import RankEstimator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %a",
)

if tf.__version__ >= "2.0":
    gfile = tf.compat.v1.gfile
else:
    gfile = tf.gfile

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_config_path", type=str, required=True)
parser.add_argument("--task_type", type=str, required=True,
                    choices=["train_and_evaluate", "evaluate", "predict", "export"])
parser.add_argument("--model_dir", type=str, required=False)
parser.add_argument("--train_input_path", type=str, required=False)
parser.add_argument("--eval_input_path", type=str, required=False)
parser.add_argument("--export_dir", type=str, required=False)
args = parser.parse_args()
print("Run params:" + str(args))


def get_pipeline_config_from_file(pipeline_config_path) -> PipelineConfig:
    with open(pipeline_config_path, "r") as f:
        res = json.load(f, object_hook=PipelineConfig.handle)
    if args.model_dir:
        res.model_dir = args.model_dir
    if args.train_input_path:
        res.input_config.train_input_path = args.train_input_path
    if args.eval_input_path:
        res.input_config.eval_input_path = args.eval_input_path
    if args.export_dir:
        res.export_config.export_dir = args.export_dir
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
    if input_config.input_type == "csv":
        input_obj = CSVInput(input_config, feature_config, input_path)
    elif input_config.input_type == "csv":
        input_obj = TFRecordInput(input_config, feature_config, input_path)
    else:
        raise ValueError("input_type:%s not supported." % input_config.input_type)
    input_fn = input_obj.create_input(export_config)
    return input_fn


def _get_ckpt_path(pipeline_config):
    ckpt_path = tf.train.latest_checkpoint(pipeline_config.model_dir)
    logging.info("use latest checkpoint %s from %s" % (ckpt_path, pipeline_config.model_dir))
    return ckpt_path


def _create_estimator(pipeline_config):
    run_config = tf.estimator.RunConfig(
        model_dir=pipeline_config.model_dir,
        log_step_count_steps=pipeline_config.train_config.log_step_count_steps,
        save_checkpoints_steps=pipeline_config.train_config.save_checkpoints_steps,
        keep_checkpoint_max=pipeline_config.train_config.keep_checkpoint_max,
    )
    estimator = RankEstimator(pipeline_config)
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
    estimator, _ = _create_estimator(pipeline_config)
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
    estimator, _ = _create_estimator(pipeline_config)
    eval_spec = _create_eval_export_spec(pipeline_config, pipeline_config.input_config.eval_input_path)

    ckpt_path = _get_ckpt_path(pipeline_config)
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
    """

    Args:
        pipeline_config:

    Returns:

    """
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
    logging.info('model has been exported to %s successfully' % final_export_dir)
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
