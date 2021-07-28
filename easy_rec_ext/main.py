# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:07 下午
# desc:

import logging

import json
import argparse
import tensorflow as tf
from easy_rec_ext.core.pipeline import PipelineConfig
from easy_rec_ext.input.csv_input import CSVInput

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
parser.add_argument("--mode", type=str, required=True, choices=["train_and_evaluate", "evaluate", "export"])
parser.add_argument("--model_dir", type=str, required=False)
parser.add_argument("--train_input_path", type=str, required=False)
parser.add_argument("--eval_input_path", type=str, required=False)
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
    return res


def get_input_fn(input_config, feature_configs, input_path=None, export_config=None):
    """Build estimator input function.

    Args:
      input_config: dataset config
      feature_configs: FeatureConfig
      input_path: input_data_path
      export_config: configuration for exporting models,
        only used to build input_fn when exporting models

    Returns:
      subclass of Input
    """
    if input_config.input_type == "csv":
        input_obj = CSVInput(input_config, feature_configs, input_path)
    else:
        raise Exception("invalid type: %s" % input_config.input_type)
    input_fn = input_obj.create_input(export_config)
    return input_fn


def train_and_evaluate(pipeline_config: PipelineConfig):
    """
    
    Args:
        pipeline_config:

    Returns:

    """
    input_config = pipeline_config.input_config
    feature_config = pipeline_config.feature_config
    train_config = pipeline_config.train_config


def evaluate(pipeline_config, eval_result_filename="eval_result.txt"):
    pass


def export(pipeline_config):
    pass


if __name__ == "__main__":
    pipeline_config = get_pipeline_config_from_file(args.pipeline_config_path)
