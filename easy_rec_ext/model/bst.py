# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 10:59 下午
# desc:

import logging
import math
import tensorflow as tf
from easy_rec_ext.layers import dnn, layer_norm
from easy_rec_ext.core import regularizers
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class BST(RankModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(BST, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0

        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._bst_tower_num = len(self._model_config.bst_towers) if self._model_config.bst_towers else 0
        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._bst_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("bst tower num: {0}".format(self._bst_tower_num))

        self._bst_tower_features = []
        for tower_id in range(self._bst_tower_num):
            tower = self._model_config.bst_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._bst_tower_features.append(tower_feature)

    def dnn_net(self, net, dnn_units, name):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            for idx, units in enumerate(dnn_units):
                net = tf.layers.dense(
                    net, units=units, activation=tf.nn.relu, name="%s_%d" % (name, idx))
        return net

    def attention_net(self, net, dim, cur_seq_len, seq_size, name):
        query_net = self.dnn_net(net, [dim], name + "_query")  # B, seq_len，dim
        key_net = self.dnn_net(net, [dim], name + "_key")
        value_net = self.dnn_net(net, [dim], name + "_value")
        scores = tf.matmul(
            query_net, key_net, transpose_b=True)  # [B, seq_size, seq_size]

        hist_mask = tf.sequence_mask(
            cur_seq_len, maxlen=seq_size - 1)  # [B, seq_size-1]
        cur_id_mask = tf.ones([tf.shape(hist_mask)[0], 1], dtype=tf.bool)  # [B, 1]
        mask = tf.concat([hist_mask, cur_id_mask], axis=1)  # [B, seq_size]
        masks = tf.reshape(tf.tile(mask, [1, seq_size]),
                           (-1, seq_size, seq_size))  # [B, seq_size, seq_size]
        padding = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(masks, scores, padding)  # [B, seq_size, seq_size]

        # Scale
        scores = tf.nn.softmax(scores)  # (B, seq_size, seq_size)
        att_res_net = tf.matmul(scores, value_net)  # [B, seq_size, emb_dim]
        return att_res_net

    def multi_head_att_net(self, id_cols, head_count, emb_dim, seq_len, seq_size):
        multi_head_attention_res = []
        part_cols_emd_dim = int(math.ceil(emb_dim / head_count))
        for start_idx in range(0, emb_dim, part_cols_emd_dim):
            if start_idx + part_cols_emd_dim > emb_dim:
                part_cols_emd_dim = emb_dim - start_idx
            part_id_col = tf.slice(id_cols, [0, 0, start_idx],
                                   [-1, -1, part_cols_emd_dim])
            part_attention_net = self.attention_net(
                part_id_col,
                part_cols_emd_dim,
                seq_len,
                seq_size,
                name="multi_head_%d" % start_idx)
            multi_head_attention_res.append(part_attention_net)
        multi_head_attention_res_net = tf.concat(multi_head_attention_res, axis=2)
        multi_head_attention_res_net = self.dnn_net(
            multi_head_attention_res_net, [emb_dim], name="multi_head_attention")
        return multi_head_attention_res_net

    def add_and_norm(self, net_1, net_2, emb_dim, name):
        net = tf.add(net_1, net_2)
        # layer = tf.keras.layers.LayerNormalization(axis=2)
        layer = layer_norm.LayerNormalization(emb_dim)
        net = layer(net)
        return net

    def bst(self, bst_fea, seq_size, head_count, name):
        cur_id, hist_id_col, seq_len = bst_fea["key"], bst_fea["hist_seq_emb"], bst_fea["hist_seq_len"]

        cur_batch_max_seq_len = tf.shape(hist_id_col)[1]

        hist_id_col = tf.cond(
            tf.constant(seq_size) > cur_batch_max_seq_len, lambda: tf.pad(
                hist_id_col, [[0, 0], [0, seq_size - cur_batch_max_seq_len - 1],
                              [0, 0]], "CONSTANT"),
            lambda: tf.slice(hist_id_col, [0, 0, 0], [-1, seq_size - 1, -1]))
        all_ids = tf.concat([hist_id_col, tf.expand_dims(cur_id, 1)],
                            axis=1)  # b, seq_size, emb_dim

        emb_dim = int(all_ids.shape[2])
        attention_net = self.multi_head_att_net(all_ids, head_count, emb_dim,
                                                seq_len, seq_size)

        tmp_net = self.add_and_norm(
            all_ids, attention_net, emb_dim, name="add_and_norm_1")
        feed_forward_net = self.dnn_net(tmp_net, [emb_dim], "feed_forward_net")
        net = self.add_and_norm(
            tmp_net, feed_forward_net, emb_dim, name="add_and_norm_2")
        bst_output = tf.reshape(net, [-1, seq_size * emb_dim])
        return bst_output

    def build_predict_graph(self):
        tower_fea_arr = []
        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_name = tower.input
            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower_name)
            tower_dnn = dnn.DNN(tower.dnn_config, self._l2_reg, "%s_dnn" % tower_name,
                                self._is_training)
            tower_fea = tower_dnn(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._bst_tower_num):
            tower_fea = self._bst_tower_features[tower_id]
            tower = self._model_config.bst_towers[tower_id]
            tower_name = tower.input
            tower_seq_len = tower.bst_config.seq_len
            tower_multi_head_size = tower.bst_config.multi_head_size
            tower_fea = self.bst(
                tower_fea,
                seq_size=tower_seq_len,
                head_count=tower_multi_head_size,
                name=tower_name)
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn",
                            self._is_training)
        all_fea = final_dnn(all_fea)

        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            all_fea = tf.concat([all_fea, bias_fea], axis=1)
        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["logits"] = logits
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
