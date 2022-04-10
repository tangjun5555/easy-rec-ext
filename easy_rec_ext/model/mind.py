# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/12/23 5:07 PM
# desc:
# MIND: Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.layers.capsule_layer import CapsuleLayer, CapsuleConfig
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.dssm import DSSMModel
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class MINDModelConfig(object):
    def __init__(self, user_input_groups: List[str], hist_input_group: str, item_input_groups: List[str],
                 capsule_config: CapsuleConfig,
                 user_field: str = None, item_field: str = None, scale_sim: bool = True,
                 ):
        self.user_input_groups = user_input_groups
        self.hist_input_group = hist_input_group
        self.item_input_groups = item_input_groups

        self.capsule_config = capsule_config

        self.user_field = user_field
        self.item_field = item_field
        self.scale_sim = scale_sim

    @staticmethod
    def handle(data):
        res = MINDModelConfig(data["user_input_groups"], data["hist_input_group"], data["item_input_groups"],
                              data["capsule_config"],
                              )
        if "user_field" in data:
            res.user_field = data["user_field"]
        if "item_field" in data:
            res.item_field = data["item_field"]
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        return res


class MINDModel(DSSMModel):
    pass


class MIND(MatchModel, MINDModel):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(MIND, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        logging.info("%s all tower num:%d" % (filename, self._dnn_tower_num))
        logging.info("%s dnn tower num:%d" % (filename, self._dnn_tower_num))

    def build_predict_graph(self):
        mind_model_config = self._model_config.mind_model_config

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

            if tower.input_group in mind_model_config.user_input_groups:
                user_emb_fea_list.append(tower_fea)
            elif tower.input_group in mind_model_config.item_input_groups:
                item_emb_fea_list.append(tower_fea)
            else:
                raise ValueError(
                    "%s tower.input_group:%s not in user or item input groups" % (filename, tower.input_group))
            logging.info("%s user_emb_fea_list add input_group:%s" % (filename, tower.input_group))
        user_emb_fea = tf.concat(user_emb_fea_list, axis=1)
        item_emb_fea = tf.concat(item_emb_fea_list, axis=1)

        user_emb_layer = dnn.DNN(self._model_config.final_dnn,
                                 self._l2_reg,
                                 "user_final_dnn",
                                 self._is_training,
                                 )
        item_emb_layer = dnn.DNN(self._model_config.final_dnn,
                                 self._l2_reg,
                                 "item_final_dnn",
                                 self._is_training,
                                 )

        hist_tower_fea_dict = self.build_seq_input_layer(mind_model_config.hist_input_group)
        hist_seq_emb, hist_seq_len = hist_tower_fea_dict["hist_seq_emb"], hist_tower_fea_dict["hist_seq_len"]

        # batch_size x max_k x high_capsule_dim
        high_capsules, num_high_capsules = CapsuleLayer("mind_capsule",
                                                        self._model_config.capsule_config,
                                                        self._is_training,
                                                        )(hist_seq_emb, hist_seq_len)

        # concatenate with user features
        user_features = tf.tile(
            tf.expand_dims(user_emb_fea, axis=1),
            [1, tf.shape(high_capsules)[1], 1],
        )
        user_features = tf.concat([high_capsules, user_features], axis=2)
        user_features = user_emb_layer(user_features)

        item_feature = item_emb_layer(item_emb_fea)
        user_features = self.norm(user_features)
        item_feature = self.norm(item_feature)
        
        # label guided attention
        # attention item features on high capsules vector
        simi = tf.einsum("bhe,be->bh", user_features, item_feature)
        simi = tf.pow(simi, self._model_config.simi_pow)
        simi_mask = tf.sequence_mask(num_high_capsules, mind_model_config.capsule_config.max_k)
        user_features = user_features * tf.to_float(simi_mask[:, :, None])

        self._prediction_dict["user_features"] = user_features
        self._prediction_dict["user_emb_num"] = num_high_capsules
        
        max_thresh = (tf.cast(simi_mask, tf.float32) * 2 - 1) * 1e32
        simi = tf.minimum(simi, max_thresh)
        simi = tf.nn.softmax(simi, axis=1)
        simi = tf.stop_gradient(simi)
        user_tower_emb = tf.einsum("bhe,bh->be", user_features, simi)

        item_tower_emb = item_feature
        user_item_sim = self.sim(user_tower_emb, item_tower_emb)

        sim_w = tf.get_variable(
            "mind/scale_sim_w",
            dtype=tf.float32,
            shape=(1, 1),
            initializer=tf.ones_initializer(),
        )
        sim_b = tf.get_variable(
            "mind/scale_sim_b",
            dtype=tf.float32,
            shape=(1,),
            initializer=tf.zeros_initializer()
        )
        tf.summary.histogram("mind/scale_sim_w", sim_w)
        tf.summary.histogram("mind/scale_sim_b", sim_b)
        user_item_sim = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
        
        probs = tf.nn.sigmoid(user_item_sim)
        self._prediction_dict["probs"] = tf.reshape(probs, (-1,), name="probs")

        self._prediction_dict["user_vector"] = tf.identity(user_tower_emb, name="user_vector")
        self._prediction_dict["item_vector"] = tf.identity(item_tower_emb, name="item_vector")

        if mind_model_config.user_field:
            self._prediction_dict["user_id"] = tf.identity(self._feature_dict[mind_model_config.user_field], name="user_id")
        if mind_model_config.item_field:
            self._prediction_dict["item_id"] = tf.identity(self._feature_dict[mind_model_config.item_field], name="item_id")
        return self._prediction_dict

    def _build_interest_metric(self):
        user_features = self._prediction_dict["user_features"]
        user_features = self.norm(user_features)
        user_feature_num = self._prediction_dict["user_emb_num"]

        user_feature_sum_sqr = tf.square(tf.reduce_sum(user_features, axis=1))
        user_feature_sqr_sum = tf.reduce_sum(tf.square(user_features), axis=1)
        simi = user_feature_sum_sqr - user_feature_sqr_sum

        simi = tf.reduce_sum(
            simi, axis=1) / tf.maximum(
            tf.to_float(user_feature_num * (user_feature_num - 1)), 1.0)
        user_feature_num = tf.reduce_sum(tf.to_float(user_feature_num > 1))
        return tf.reduce_sum(simi) / tf.maximum(user_feature_num, 1.0)

    def build_metric_graph(self, eval_config):
        metric_dict = super(MIND, self).build_metric_graph(eval_config)
        metric_dict["interest_similarity"] = self._build_interest_metric()
        return metric_dict
