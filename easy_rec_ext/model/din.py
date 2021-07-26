# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 2:14 下午
# desc:

import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.core.pipeline import PipelineConfig
from easy_rec_ext.model.rank_model import RankModel


class DINConfig(dnn.DNNConfig):
  def __init__(self, hidden_units: List[int] = (64, 32, 1), activation: str = "tf.nn.relu", use_bn: bool = False,
               dropout_ratio=None):
    super().__init__(hidden_units, activation, use_bn, dropout_ratio)
    assert self.hidden_units[-1] == 1


class DIN(RankModel):
  def __init__(self, pipeline_config: PipelineConfig):
    super(DIN, self).__init__(pipeline_config=pipeline_config)

  def din(self, dnn_config, deep_fea, name):
    cur_id, hist_id_col, seq_len = deep_fea['key'], deep_fea[
        'hist_seq_emb'], deep_fea['hist_seq_len']

    seq_max_len = tf.shape(hist_id_col)[1]
    emb_dim = hist_id_col.shape[2]

    cur_ids = tf.tile(cur_id, [1, seq_max_len])
    cur_ids = tf.reshape(cur_ids,
                         tf.shape(hist_id_col))  # (B, seq_max_len, emb_dim)

    din_net = tf.concat(
        [cur_ids, hist_id_col, cur_ids - hist_id_col, cur_ids * hist_id_col],
        axis=-1)  # (B, seq_max_len, emb_dim*4)

    din_layer = dnn.DNN(dnn_config, self._l2_reg, name, self._is_training)
    din_net = din_layer(din_net)
    scores = tf.reshape(din_net, [-1, 1, seq_max_len])  # (B, 1, ?)

    seq_len = tf.expand_dims(seq_len, 1)
    mask = tf.sequence_mask(seq_len)
    padding = tf.ones_like(scores) * (-2**32 + 1)
    scores = tf.where(mask, scores, padding)  # [B, 1, seq_max_len]

    # Scale
    scores = tf.nn.softmax(scores)  # (B, 1, seq_max_len)
    hist_din_emb = tf.matmul(scores, hist_id_col)  # [B, 1, emb_dim]
    hist_din_emb = tf.reshape(hist_din_emb, [-1, emb_dim])  # [B, emb_dim]
    din_output = tf.concat([hist_din_emb, cur_id], axis=1)
    return din_output

  def _model_fn(self, features, labels, mode, config, params):
    pass
