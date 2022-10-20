# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/10/19 16:49
# desc:

import logging
import tensorflow as tf
from easy_rec_ext.layers import dnn, interaction
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class XDeepFMConfig(object):
    def __init__(self, layer_size=(128, 128)):
        self.layer_size = layer_size

    @staticmethod
    def handle(data):
        res = XDeepFMConfig()
        if "layer_size" in data:
            res.layer_size = data["layer_size"]
        return res


class XDeepFMTower(object):
    def __init__(self, input_group, xdeepfm_config: XDeepFMConfig):
        self.input_group = input_group
        self.xdeepfm_config = xdeepfm_config

    @staticmethod
    def handle(data):
        xdeepfm_config = XDeepFMConfig.handle(data["xdeepfm_config"])
        res = XDeepFMTower(data["input_group"], xdeepfm_config)
        return res


class XDeepFMLayer(object):
    def call(self, name: str, xdeepfm_config: XDeepFMConfig, deep_fea: tf.Tensor):
        output = interaction.CIN(
            name=name + "_cin",
            layer_size=xdeepfm_config.layer_size,
        )(deep_fea)
        logging.info("XDeepFMLayer %s, output.shape:%s" % (name, str(output.shape)))
        return output


class XDeepFM(RankModel, XDeepFMLayer):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False):
        super(XDeepFM, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._xdeepfm_tower_num = len(self._model_config.xdeepfm_towers) if self._model_config.xdeepfm_towers else 0
        self._xdeepfm_tower_features = []
        for tower_id in range(self._xdeepfm_tower_num):
            tower = self._model_config.xdeepfm_towers[tower_id]
            tower_feature = self.build_interaction_input_layer(tower.input_group)
            self._xdeepfm_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._xdeepfm_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("xdeepfm tower num: {0}".format(self._xdeepfm_tower_num))

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

        for tower_id in range(self._xdeepfm_tower_num):
            tower_fea = self._xdeepfm_tower_features[tower_id]
            tower = self._model_config.fibitnet_towers[tower_id]
            tower_fea = self.call(
                name=tower.input_group + "_xdeepfm",
                xdeepfm_config=tower.xdeepfm_config,
                deep_fea=tower_fea,
            )
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        if self._model_config.final_dnn:
            final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
            all_fea = final_dnn_layer(all_fea)
        logging.info("XDeepFM build_predict_graph, all_fea.shape:%s" % (str(all_fea.shape)))

        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
