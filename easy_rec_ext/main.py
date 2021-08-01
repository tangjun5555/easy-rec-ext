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


def _create_eval_export_spec(pipeline_config, eval_data):
    # data_config = pipeline_config.data_config
    # feature_configs = pipeline_config.feature_configs
    # eval_config = pipeline_config.eval_config
    # export_config = pipeline_config.export_config

    # if eval_config.num_examples > 0:
    #     eval_steps = int(
    #         math.ceil(float(eval_config.num_examples) / data_config.batch_size))
    #     logging.info('eval_steps = %d' % eval_steps)
    # else:
    #     eval_steps = None
    # create eval input

    export_input_fn = get_input_fn(pipeline_config.input_config,
                                   pipeline_config.feature_config,
                                   None,
                                   pipeline_config.export_config
                                   )
    exporters = [
        FinalExporter(name='final', serving_input_receiver_fn=export_input_fn)
    ]

    # if export_config.exporter_type == 'final':
    #     exporters = [
    #         FinalExporter(name='final', serving_input_receiver_fn=export_input_fn)
    #     ]
    # elif export_config.exporter_type == 'latest':
    #     exporters = [
    #         LatestExporter(
    #             name='latest',
    #             serving_input_receiver_fn=export_input_fn,
    #             exports_to_keep=export_config.exports_to_keep)
    #     ]
    # elif export_config.exporter_type == 'best':
    #     logging.info(
    #         'will use BestExporter, metric is %s, the bigger the better: %d' %
    #         (export_config.best_exporter_metric, export_config.metric_bigger))
    #
    #     def _metric_cmp_fn(best_eval_result, current_eval_result):
    #         logging.info('metric: best = %s current = %s' %
    #                      (str(best_eval_result), str(current_eval_result)))
    #         if export_config.metric_bigger:
    #             return (best_eval_result[export_config.best_exporter_metric] <
    #                     current_eval_result[export_config.best_exporter_metric])
    #         else:
    #             return (best_eval_result[export_config.best_exporter_metric] >
    #                     current_eval_result[export_config.best_exporter_metric])
    #
    #     exporters = [
    #         BestExporter(
    #             name='best',
    #             serving_input_receiver_fn=export_input_fn,
    #             compare_fn=_metric_cmp_fn,
    #             exports_to_keep=export_config.exports_to_keep)
    #     ]
    # elif export_config.exporter_type == 'none':
    #     exporters = []
    # else:
    #     raise ValueError('Unknown exporter type %s' % export_config.exporter_type)

    # set throttle_secs to a small number, so that we can control evaluation
    # interval steps by checkpoint saving steps
    eval_input_fn = get_input_fn(pipeline_config.input_config, pipeline_config.feature_config, eval_data)
    eval_spec = tf.estimator.EvalSpec(
        name='val',
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
    train_input_fn = get_input_fn(pipeline_config.input_config,
                                  pipeline_config.feature_config,
                                  pipeline_config.input_config.train_input_path
                                  )
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = _create_eval_export_spec(pipeline_config, pipeline_config.input_config.eval_input_path)


def evaluate(pipeline_config, eval_result_filename="eval_result.txt"):
    pass


def export(pipeline_config):
    pass


if __name__ == "__main__":
    pipeline_config = get_pipeline_config_from_file(args.pipeline_config_path)
