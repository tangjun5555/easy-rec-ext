# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/12 5:05 下午
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.layers.interaction import FM
from easy_rec_ext.core import regularizers
from easy_rec_ext.model.rank_model import RankModel
from easy_rec_ext.model.din import DINLayer
from easy_rec_ext.model.bst import BSTLayer

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


filename = str(os.path.basename(__file__)).split(".")[0]


class MultiTower(RankModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False):
        super(MultiTower, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._interaction_tower_num = len(self._model_config.interaction_towers) if self._model_config.interaction_towers else 0
        self._interaction_tower_features = []
        for tower_id in range(self._interaction_tower_num):
            tower = self._model_config.interaction_towers[tower_id]
            tower_feature = self.build_id_feature_input_layer(tower.input_group)
            self._interaction_tower_features.append(tower_feature)

        self._din_tower_num = len(self._model_config.din_towers)
        self._din_tower_features = []
        for tower_id in range(self._din_tower_num):
            tower = self._model_config.din_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._din_tower_features.append(tower_feature)

        self._bst_tower_num = len(self._model_config.bst_towers) if self._model_config.bst_towers else 0
        self._bst_tower_features = []
        for tower_id in range(self._bst_tower_num):
            tower = self._model_config.bst_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._bst_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._interaction_tower_num + self._din_tower_num + self._bst_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("interaction tower num: {0}".format(self._interaction_tower_num))
        logging.info("din tower num: {0}".format(self._din_tower_num))
        logging.info("bst tower num: {0}".format(self._bst_tower_num))

    def build_tower_fea_arr(self):
        tower_fea_arr = []

        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_name = tower.input_group
            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower_name
            )
            dnn_layer = dnn.DNN(tower.dnn_config, self._l2_reg, "%s_dnn" % tower_name, self._is_training)
            tower_fea = dnn_layer(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._interaction_tower_num):
            tower_fea = self._interaction_tower_features[tower_id]
            tower = self._model_config.interaction_towers[tower_id]
            tower_name = tower.input_group

            if tower.interaction_config.mode == "fm":
                fm_layer = FM(tower_name + "_fm")
                tower_fea_arr.append(fm_layer(tower_fea))

        for tower_id in range(self._din_tower_num):
            tower_fea = self._din_tower_features[tower_id]
            tower = self._model_config.din_towers[tower_id]
            din_layer = DINLayer()
            tower_fea = din_layer.din(tower.din_config, tower_fea, name="%s_din" % tower.input_group)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._bst_tower_num):
            tower_fea = self._bst_tower_features[tower_id]
            tower = self._model_config.bst_towers[tower_id]
            bst_layer = BSTLayer()
            tower_fea = bst_layer.bst(
                tower_fea,
                seq_size=tower.bst_config.seq_size,
                head_count=tower.bst_config.multi_head_size,
                name=tower.input_group,
            )
            tower_fea_arr.append(tower_fea)

        return tower_fea_arr

    def build_predict_graph(self):
        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, all_fea.length:%s" % (filename, str(len(tower_fea_arr))))

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn_layer = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
        all_fea = final_dnn_layer(all_fea)
        logging.info("build_predict_graph, logits.shape:%s" % (str(all_fea.shape)))

        if self._model_config.bias_tower:
            bias_fea = self.build_bias_input_layer(self._model_config.bias_tower.input_group)
            all_fea = tf.concat([all_fea, bias_fea], axis=1)
            logging.info("build_predict_graph, logits.shape:%s" % (str(all_fea.shape)))
        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["logits"] = logits
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
