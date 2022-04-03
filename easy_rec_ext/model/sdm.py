# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/19 6:35 下午
# desc:
# Reference:
# SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.dssm import DSSMModel, DSSMModelConfig
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class SDMModelConfig(DSSMModelConfig):
    def __init__(self, user_input_groups: List[str], item_input_groups: List[str],
                 user_field: str, item_field: str,
                 scale_sim: bool = True, use_user_scale_weight=False, use_item_scale_weight=False,):
        super(SDMModelConfig, self).__init__(
            user_input_groups, item_input_groups,
            user_field, item_field,
            scale_sim, use_user_scale_weight, use_item_scale_weight
        )

    @staticmethod
    def handle(data):
        res = SDMModelConfig(data["user_input_groups"], data["item_input_groups"], data["user_field"],
                              data["item_field"])
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        if "use_user_scale_weight" in data:
            res.user_size = data["use_user_scale_weight"]
        if "use_item_scale_weight" in data:
            res.item_size = data["use_item_scale_weight"]
        return res


class SDMModel(DSSMModel):
    def sdm(self, name, short_hist, long_hist):
        pass


class SDM(MatchModel, SDMModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(SDM, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        logging.info("%s all tower num:%d" % (filename, self._dnn_tower_num))
        logging.info("%s dnn tower num:%d" % (filename, self._dnn_tower_num))
