# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/30 12:00 PM
# desc: Co-Action Network

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.core import regularizers
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class CANConfig(object):
    def __init__(self, mlp_units: List[int], item_vocab_size):
        self.mlp_units = mlp_units
        self.item_vocab_size = item_vocab_size

    @staticmethod
    def handle(data):
        res = CANConfig(data["mlp_units"], data["item_vocab_size"])
        return res

    def __str__(self):
        return str(self.__dict__)


class CANTower(object):
    def __init__(self, input_group, can_config: CANConfig):
        self.input_group = input_group
        self.can_config = can_config

    @staticmethod
    def handle(data):
        res = CANTower(data["input_group"], CANConfig.handle(data["can_config"]))
        return res

    def __str__(self):
        return str(self.__dict__)


class CANLayer(object):
    def call(self, name, deep_fea, item_vocab_size, mlp_units):
        """
        Args:
            name:
            deep_fea: dict
                user_value:
                    3D tensor with shape: (batch_size, field_num, embedding_size)
                    fed into MLP
                item_value:
                    2D tensor with shape: (batch_size, 1), use the original id
                    server as the weight and bias
            item_vocab_size:
            mlp_units:
        Returns:
            2D tensor with shape: (batch_size, sum(mlp_units))
        """
        user_value, item_value = deep_fea["user_value"], deep_fea["item_value"]
        user_dimension = user_value.get_shape().as_list()[-1]
        order = len(mlp_units)

        item_dimension = 0
        weight_emb_w = []
        weight_emb_b = []
        x = user_dimension
        for y in mlp_units:
            item_dimension += x * y
            item_dimension += 1
            weight_emb_w.append([x, y])
            weight_emb_b.append(1)
            x = y

        item_embedding_weight = embedding_ops.get_embedding_variable(
            name=name + "_item",
            dim=item_dimension,
            vocab_size=item_vocab_size
        )
        item_embedding_value = embedding_ops.safe_embedding_lookup(
            item_embedding_weight, item_value,
        )

        weight, bias = [], []
        idx = 0
        weight_orders = []
        bias_orders = []
        for i in range(order):
            for w, b in zip(weight_emb_w, weight_emb_b):
                weight.append(tf.slice(item_embedding_value, [0, idx], [-1, w[0] * w[1]]))
                idx += w[0] * w[1]
                bias.append(tf.slice(item_embedding_value, [0, idx], [-1, b]))
                idx += b
            weight_orders.append(weight)
            bias_orders.append(bias)

        hh = [user_value]
        for i in range(order - 1):
            hh.append(tf.multiply(hh[-1] ** user_value))

        out_seq = []
        for i, h in enumerate(hh):
            weight, bias = weight_orders[i], bias_orders[i]
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)
                h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)
            out_seq.append(h)

        out_seq = tf.concat(out_seq, 2)
        return tf.reduce_sum(out_seq, 1, keepdims=False)


class CAN(RankModel, CANLayer):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(CAN, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._can_tower_num = len(self._model_config.can_towers) if self._model_config.can_towers else 0
        self._can_tower_features = []
        for tower_id in range(self._can_tower_num):
            tower = self._model_config.can_towers[tower_id]
            tower_feature = self.build_cartesian_interaction_input_layer(tower.input_group, True)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["user_value"]])
            self._can_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._can_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("can tower num: {0}".format(self._can_tower_num))

    def build_predict_graph(self):
        tower_fea_arr = []

        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_name = tower.input_group
            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower_name,
            )
            dnn_layer = dnn.DNN(
                tower.dnn_config,
                self._l2_reg,
                "%s_dnn" % tower_name,
                self._is_training,
            )
            tower_fea = dnn_layer(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._can_tower_num):
            tower_fea = self._can_tower_features[tower_id]
            tower = self._model_config.can_towers[tower_id]
            tower_fea = self.call(
                name="%s_can" % tower.input_group,
                deep_fea=tower_fea,
                item_vocab_size=tower.can_config.item_vocab_size,
                mlp_units=tower.can_config.mlp_units,
            )
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
        all_fea = final_dnn_layer(all_fea)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            all_fea = tf.concat([all_fea, bias_fea], axis=1)
            logging.info("build_predict_graph, all_fea.shape:%s" % (str(all_fea.shape)))

        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
