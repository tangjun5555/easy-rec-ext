# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/10/12 16:52
# desc:

import logging

import tensorflow as tf
from easy_rec_ext.layers import dnn, senet, interaction
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class FiBiNetConfig(object):
    def __init__(self, senet_reduction_ratio=2.0, bilinear_type="Field-Interaction"):
        self.senet_reduction_ratio = senet_reduction_ratio
        self.bilinear_type = bilinear_type

    @staticmethod
    def handle(data):
        res = FiBiNetConfig()
        if "senet_reduction_ratio" in data:
            res.senet_reduction_ratio = data["senet_reduction_ratio"]
        if "bilinear_type" in data:
            res.bilinear_type = data["bilinear_type"]
        return res


class FiBiNetTower(object):
    def __init__(self, input_group, fibinet_config: FiBiNetConfig):
        self.input_group = input_group
        self.fibinet_config = fibinet_config

    @staticmethod
    def handle(data):
        fibinet_config = FiBiNetConfig.handle(data["fibinet_config"])
        res = FiBiNetTower(data["input_group"], fibinet_config)
        return res


class FiBiNetLayer(object):
    def call(self, name: str, fibinet_config: FiBiNetConfig, deep_fea: tf.Tensor):
        senet_embedding = senet.SENetLayer(
            name=name + "_senet",
            reduction_ratio=fibinet_config.senet_reduction_ratio,
        )(deep_fea)
        senet_bilinear_out = interaction.BilinearInteraction(
            name=name + "_senet_bilinear",
            bilinear_type=fibinet_config.bilinear_type,
        )(senet_embedding)
        bilinear_out = interaction.BilinearInteraction(
            name=name + "_bilinear",
            bilinear_type=fibinet_config.bilinear_type,
        )(deep_fea)
        output = tf.concat([senet_bilinear_out, bilinear_out], axis=-1)
        logging.info("FiBiNetLayer %s, output.shape:%s" % (name, str(output.shape)))
        return output


class FiBiNet(RankModel, FiBiNetLayer):

    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False):
        super(FiBiNet, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._fibitnet_tower_num = len(self._model_config.fibitnet_towers) if self._model_config.fibitnet_towers else 0
        self._fibitnet_tower_features = []
        for tower_id in range(self._fibitnet_tower_num):
            tower = self._model_config.fibitnet_towers[tower_id]
            tower_feature = self.build_interaction_input_layer(tower.input_group)
            self._fibitnet_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._fibitnet_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("fibinet tower num: {0}".format(self._fibitnet_tower_num))

    def build_predict_graph(self):
        tower_fea_arr = []

        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower.input_group)
            dnn_layer = dnn.DNN(tower.dnn_config, self._l2_reg, "%s_dnn" % tower.input_group,
                                self._is_training)
            tower_fea = dnn_layer(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._fibitnet_tower_num):
            tower_fea = self._fibitnet_tower_features[tower_id]
            tower = self._model_config.fibitnet_towers[tower_id]
            tower_fea = self.call(
                name=tower.input_group + "_fibinet",
                fibinet_config=tower.fibinet_config,
                deep_fea=tower_fea,
            )
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
        all_fea = final_dnn_layer(all_fea)
        logging.info("FiBiNet build_predict_graph, all_fea.shape:%s" % (str(all_fea.shape)))

        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            all_fea = tf.concat([all_fea, bias_fea], axis=1)
            logging.info("FiBiNet build_predict_graph, all_fea.shape:%s" % (str(all_fea.shape)))

        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
