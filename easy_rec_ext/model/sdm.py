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
    def __init__(self, user_input_groups: List[str], hist_long_input_group: str, hist_short_input_group: str,
                 item_input_groups: List[str],
                 user_field: str = None, item_field: str = None, scale_sim: bool = True,
                 ):
        self.user_input_groups = user_input_groups
        self.hist_long_input_group = hist_long_input_group
        self.hist_short_input_group = hist_short_input_group
        self.item_input_groups = item_input_groups

        self.user_field = user_field
        self.item_field = item_field
        self.scale_sim = scale_sim

    @staticmethod
    def handle(data):
        res = SDMModelConfig(data["user_input_groups"], data["hist_long_input_group"], data["hist_short_input_group"],
                             data["item_input_groups"])
        if "user_field" in data:
            res.user_field = data["user_field"]
        if "item_field" in data:
            res.item_field = data["item_field"]
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        return res


class SDMModel(DSSMModel):
    def sdm(self, name, hist_long_feas, hist_short_feas, user_profile_emd):
        hist_short_seq_emb, hist_short_seq_len = hist_short_feas["hist_seq_emb"], hist_short_feas["hist_seq_len"]


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
