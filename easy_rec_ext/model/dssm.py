# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/19 6:35 下午
# desc:

import os
import logging
from typing import List
import tensorflow as tf
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.layers import dnn
from easy_rec_ext.model.match_model import MatchModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class DSSMModelConfig(object):
    def __init__(self, user_input_groups: List[str], item_input_groups: List[str],
                 user_field: str, item_field: str,
                 scale_sim: bool = True, use_user_scale_weight=False, use_item_scale_weight=False,
                 ):
        self.user_input_groups = user_input_groups
        self.item_input_groups = item_input_groups
        self.user_field = user_field
        self.item_field = item_field
        self.scale_sim = scale_sim
        self.use_user_scale_weight = use_user_scale_weight
        self.use_item_scale_weight = use_item_scale_weight

    @staticmethod
    def handle(data):
        res = DSSMModelConfig(data["user_input_groups"], data["item_input_groups"], data["user_field"],
                              data["item_field"])
        if "scale_sim" in data:
            res.scale_sim = data["scale_sim"]
        if "use_user_scale_weight" in data:
            res.user_size = data["use_user_scale_weight"]
        if "use_item_scale_weight" in data:
            res.item_size = data["use_item_scale_weight"]
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

        logging.info("%s all tower num:%d" % (filename, self._dnn_tower_num))
        logging.info("%s dnn tower num:%d" % (filename, self._dnn_tower_num))

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
            if dssm_model_config.use_user_scale_weight:
                user_feature_field = self._feature_fields_dict[dssm_model_config.user_field]
                user_id = self._feature_dict[dssm_model_config.user_field]
                embedding_weights = embedding_ops.get_embedding_variable(
                    name="scale_sim_w",
                    dim=1,
                    vocab_size=user_feature_field.num_buckets if user_feature_field.num_buckets > 0 else user_feature_field.hash_bucket_size,
                    key_is_string=False,
                )
                sim_w = embedding_ops.safe_embedding_lookup(
                    embedding_weights, user_id
                )
            else:
                sim_w = tf.get_variable(
                    "scale_sim_w",
                    dtype=tf.float32,
                    shape=(1, 1),
                    initializer=tf.ones_initializer(),
                )

            if dssm_model_config.use_item_scale_weight:
                item_feature_field = self._feature_fields_dict[dssm_model_config.item_field]
                item_id = self._feature_dict[dssm_model_config.item_field]
                embedding_weights = embedding_ops.get_embedding_variable(
                    name="scale_sim_b",
                    dim=1,
                    vocab_size=item_feature_field.num_buckets if item_feature_field.num_buckets > 0 else item_feature_field.hash_bucket_size,
                    key_is_string=False,
                )
                sim_b = embedding_ops.safe_embedding_lookup(
                    embedding_weights, item_id
                )
                sim_b = tf.squeeze(sim_b, axis=-1)
            else:
                sim_b = tf.get_variable(
                    "scale_sim_b",
                    dtype=tf.float32,
                    shape=(1,),
                    initializer=tf.zeros_initializer()
                )

            user_item_sim = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b

        user_item_sim = tf.nn.sigmoid(user_item_sim)
        user_item_sim = tf.reshape(user_item_sim, (-1,))

        prediction_dict = dict()
        prediction_dict["probs"] = user_item_sim

        prediction_dict["user_id"] = self._feature_dict[dssm_model_config.user_field]
        prediction_dict["user_emb"] = tf.reduce_join(
            tf.as_string(user_emb),
            axis=-1,
            separator=",",
        )
        prediction_dict["item_id"] = self._feature_dict[dssm_model_config.item_field]
        prediction_dict["item_emb"] = tf.reduce_join(
            tf.as_string(item_emb),
            axis=-1,
            separator=",",
        )
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
