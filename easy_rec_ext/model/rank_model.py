# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:34 下午
# desc:

from abc import abstractmethod
from easy_rec_ext.core.pipeline import PipelineConfig
import tensorflow as tf


class RankModel(tf.estimator.Estimator):
  def __init__(self, pipeline_config: PipelineConfig):
    super(RankModel, self).__init__(
      model_fn=self._model_fn,
      model_dir=pipeline_config.model_dir,
      config=None,
      params=None,
      warm_start_from=None
    )

  @abstractmethod
  def _model_fn(self, features, labels, mode, config, params):
    pass
