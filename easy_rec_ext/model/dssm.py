# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/19 6:35 下午
# desc:

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.layers import sequence_pooling
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class DSSMModelConfig(object):
    def __init__(self, user_input_groups: List[str], item_input_groups: List[str],
                 user_field: str = None, item_field: str = None,
                 scale_sim: bool = True,
                 ):
        self.user_input_groups = user_input_groups
        self.item_input_groups = item_input_groups
        self.user_field = user_field
        self.item_field = item_field
        self.scale_sim = scale_sim

    @staticmethod
    def handle(data):
        res = DSSMModelConfig(data["user_input_groups"], data["item_input_groups"])
        if "user_field" in data:
            res.user_field = data["user_field"]
        if "item_field" in data:
            res.item_field = data["item_field"]
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        return res

    def __str__(self):
        return str(self.__dict__)


class DSSMModel(object):
    def norm(self, fea):
        fea_norm = tf.nn.l2_normalize(fea, axis=1)
        return fea_norm

    def point_wise_sim(self, user_emb, item_emb):
        user_item_sim = tf.reduce_sum(
            tf.multiply(user_emb, item_emb), axis=1, keep_dims=True
        )
        return user_item_sim

    def list_wise_sim(self, user_emb, item_emb, hard_neg_indices=None):
        batch_size = tf.shape(user_emb)[0]

        if hard_neg_indices is not None:
            logging.info("%s list_wise_sim with hard negative examples" % (filename))
            noclk_size = tf.shape(hard_neg_indices)[0]
            pos_item_emb, neg_item_emb, hard_neg_item_emb = tf.split(
                item_emb, [batch_size, -1, noclk_size], axis=0,
            )
        else:
            pos_item_emb = item_emb[:batch_size]
            neg_item_emb = item_emb[batch_size:]

        pos_user_item_sim = tf.reduce_sum(
            tf.multiply(user_emb, pos_item_emb), axis=1, keep_dims=True
        )
        neg_user_item_sim = tf.matmul(user_emb, tf.transpose(neg_item_emb))

        if hard_neg_indices is not None:
            user_emb_expand = tf.gather(user_emb, hard_neg_indices[:, 0])
            hard_neg_user_item_sim = tf.reduce_sum(
                tf.multiply(user_emb_expand, hard_neg_item_emb), axis=1
            )
            # scatter hard negatives sim update neg_user_item_sim
            neg_sim_shape = tf.shape(neg_user_item_sim, out_type=tf.int64)
            hard_neg_mask = tf.scatter_nd(
                hard_neg_indices,
                tf.ones_like(hard_neg_user_item_sim, dtype=tf.bool),
                shape=neg_sim_shape
            )
            hard_neg_user_item_sim = tf.scatter_nd(
                hard_neg_indices, hard_neg_user_item_sim, shape=neg_sim_shape
            )
            neg_user_item_sim = tf.where(
                hard_neg_mask, x=hard_neg_user_item_sim, y=neg_user_item_sim
            )

        user_item_sim = tf.concat([pos_user_item_sim, neg_user_item_sim], axis=1)
        return user_item_sim

    def sim(self, user_emb, item_emb, is_point_wise=True):
        if is_point_wise:
            return self.point_wise_sim(user_emb, item_emb)
        else:
            return self.list_wise_sim(user_emb, item_emb)


class DSSM(MatchModel, DSSMModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(DSSM, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._seq_pooling_tower_num = len(self._model_config.seq_pooling_towers) if self._model_config.seq_pooling_towers else 0
        self._seq_pooling_tower_features = []
        for tower_id in range(self._seq_pooling_tower_num):
            tower = self._model_config.seq_pooling_towers[tower_id]
            tower_feature = self.build_seq_input_layer(tower.input_group)
            self._seq_pooling_tower_features.append(tower_feature)

        logging.info("%s all tower num:%d" % (filename, self._dnn_tower_num + self._seq_pooling_tower_num))
        logging.info("%s dnn tower num:%d" % (filename, self._dnn_tower_num))
        logging.info("%s seq_pooling tower num:%d" % (filename, self._seq_pooling_tower_num))

    def build_predict_graph(self):
        dssm_model_config = self._model_config.dssm_model_config

        user_emb_fea_list = []
        item_emb_fea_list = []

        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_fea = self._dnn_tower_features[tower_id]

            tower_fea = tf.layers.batch_normalization(
                tower_fea,
                training=self._is_training,
                trainable=True,
                name="%s_fea_bn" % tower.input_group,
            )
            tower_fea = dnn.DNN(
                tower.dnn_config,
                self._l2_reg,
                "%s_dnn" % tower.input_group,
                self._is_training
            )(tower_fea)

            if tower.input_group in dssm_model_config.user_input_groups:
                user_emb_fea_list.append(tower_fea)
            elif tower.input_group in dssm_model_config.item_input_groups:
                item_emb_fea_list.append(tower_fea)
            else:
                raise ValueError("tower.input_group:%s not in user or item input groups" % tower.input_group)
            logging.info("DSSM user_emb_fea_list add input_group:%s" % tower.input_group)

        for tower_id in range(self._seq_pooling_tower_num):
            tower = self._model_config.seq_pooling_towers[tower_id]
            tower_fea = self._seq_pooling_tower_features[tower_id]
            tower_fea = sequence_pooling.SequencePooling(
                name=tower.input_group + "_pooling",
                mode=tower.sequence_pooling_config.mode,
                gru_config=tower.sequence_pooling_config.gru_config,
                lstm_config=tower.sequence_pooling_config.lstm_config,
                self_att_config=tower.sequence_pooling_config.self_att_config,
            )(tower_fea["hist_seq_emb"], tower_fea["hist_seq_len"])

            if tower.input_group in dssm_model_config.user_input_groups:
                user_emb_fea_list.append(tower_fea)
            elif tower.input_group in dssm_model_config.item_input_groups:
                item_emb_fea_list.append(tower_fea)
            else:
                raise ValueError("tower.input_group:%s not in user or item input groups" % tower.input_group)
            logging.info("DSSM user_emb_fea_list add input_group:%s" % tower.input_group)

        user_emb_fea = tf.concat(user_emb_fea_list, axis=1)
        item_emb_fea = tf.concat(item_emb_fea_list, axis=1)

        user_emb = dnn.DNN(self._model_config.final_dnn,
                           self._l2_reg,
                           "user_final_dnn",
                           self._is_training
                           )(user_emb_fea)
        item_emb = dnn.DNN(self._model_config.final_dnn,
                           self._l2_reg,
                           "item_final_dnn",
                           self._is_training
                           )(item_emb_fea)

        user_emb = self.norm(user_emb)
        item_emb = self.norm(item_emb)
        user_item_sim = self.sim(user_emb, item_emb)

        if dssm_model_config.scale_sim:
            sim_w = tf.get_variable(
                "dssm/scale_sim_w",
                dtype=tf.float32,
                shape=(1, 1),
                initializer=tf.ones_initializer(),
            )
            sim_b = tf.get_variable(
                "dssm/scale_sim_b",
                dtype=tf.float32,
                shape=(1,),
                initializer=tf.zeros_initializer()
            )
            tf.summary.histogram("dssm/scale_sim_w", sim_w)
            tf.summary.histogram("dssm/scale_sim_b", sim_b)
            user_item_sim = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
        probs = tf.nn.sigmoid(user_item_sim)

        self._prediction_dict["probs"] = tf.reshape(probs, (-1,), name="probs")

        self._prediction_dict["user_vector"] = tf.identity(user_emb, name="user_vector")
        self._prediction_dict["item_vector"] = tf.identity(item_emb, name="item_vector")

        if dssm_model_config.user_field:
            self._prediction_dict["user_id"] = tf.identity(self._feature_dict[dssm_model_config.user_field], name="user_id")
        if dssm_model_config.item_field:
            self._prediction_dict["item_id"] = tf.identity(self._feature_dict[dssm_model_config.item_field], name="item_id")

        return self._prediction_dict
