# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/12 5:05 下午
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.layers import dnn
from easy_rec_ext.layers.interaction import FM, InnerProduct, OuterProduct, BilinearInteraction
from easy_rec_ext.core import regularizers
from easy_rec_ext.model.rank_model import RankModel
from easy_rec_ext.model.din import DINLayer
from easy_rec_ext.model.bst import BSTLayer
from easy_rec_ext.model.dien import DIENLayer
from easy_rec_ext.model.can import CANLayer
from easy_rec_ext.model.star import StarTopologyFCNLayer, AuxiliaryNetworkLayer as STARAuxiliaryNetworkLayer

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

        self._wide_tower_num = len(self._model_config.wide_towers) if self._model_config.wide_towers else 0
        self._wide_tower_features = []
        for tower_id in range(self._wide_tower_num):
            tower = self._model_config.wide_towers[tower_id]
            tower_feature = self.build_wide_input_layer(tower)
            self._wide_tower_features.append(tower_feature)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._interaction_tower_num = len(
            self._model_config.interaction_towers) if self._model_config.interaction_towers else 0
        self._interaction_tower_features = []
        for tower_id in range(self._interaction_tower_num):
            tower = self._model_config.interaction_towers[tower_id]
            tower_feature = self.build_interaction_input_layer(tower.input_group)
            self._interaction_tower_features.append(tower_feature)

        self._din_tower_num = len(self._model_config.din_towers) if self._model_config.din_towers else 0
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

        self._dien_tower_num = len(self._model_config.dien_towers) if self._model_config.dien_towers else 0
        self._dien_tower_features = []
        for tower_id in range(self._dien_tower_num):
            tower = self._model_config.dien_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._dien_tower_features.append(tower_feature)

        self._can_tower_num = len(self._model_config.can_towers) if self._model_config.can_towers else 0
        self._can_tower_features = []
        for tower_id in range(self._can_tower_num):
            tower = self._model_config.can_towers[tower_id]
            tower_feature = self.build_cartesian_interaction_input_layer(tower.input_group, True)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["user_value"]])
            self._can_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(
            self._wide_tower_num
            + self._dnn_tower_num
            + self._interaction_tower_num
            + self._din_tower_num
            + self._bst_tower_num
            + self._dien_tower_num
            + self._can_tower_num
        ))
        logging.info("wide tower num: {0}".format(self._wide_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("interaction tower num: {0}".format(self._interaction_tower_num))
        logging.info("din tower num: {0}".format(self._din_tower_num))
        logging.info("bst tower num: {0}".format(self._bst_tower_num))
        logging.info("dien tower num: {0}".format(self._dien_tower_num))
        logging.info("can tower num: {0}".format(self._can_tower_num))

    def build_tower_fea_arr(self, variable_scope=None):
        tower_fea_arr = []
        variable_scope = variable_scope if variable_scope else "multi_tower"

        for tower_id in range(self._dnn_tower_num):
            tower_fea = self._dnn_tower_features[tower_id]
            tower = self._model_config.dnn_towers[tower_id]
            tower_name = tower.input_group

            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
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

            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
                if tower.interaction_config.mode == "fm":
                    tower_fea_arr.append(FM(tower_name + "_" + "fm")(tower_fea))
                elif tower.interaction_config.mode in ["inner_product", "InnerProduct"]:
                    tower_fea_arr.append(InnerProduct(tower_name + "_" + "inner_product")(tower_fea))
                elif tower.interaction_config.mode in ["outer_product", "OuterProduct"]:
                    tower_fea_arr.append(OuterProduct(tower_name + "_" + "outer_product")(tower_fea))
                elif tower.interaction_config.mode in ["bilinear_interaction", "BilinearInteraction"]:
                    tower_fea_arr.append(BilinearInteraction(tower_name + "_" + "bilinear_interaction")(tower_fea))
                else:
                    raise ValueError(
                        "%s interaction_config.mode:%s is not supported." % (filename, tower.interaction_config.mode)
                    )

        for tower_id in range(self._din_tower_num):
            tower_fea = self._din_tower_features[tower_id]
            tower = self._model_config.din_towers[tower_id]

            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
                din_layer = DINLayer()
                tower_fea = din_layer.din(
                    tower.din_config.dnn_config,
                    tower_fea,
                    name="%s_din" % tower.input_group,
                    return_target=tower.din_config.return_target,
                    limit_seq_size=tower.din_config.limit_seq_size,
                )
                tower_fea_arr.append(tower_fea)

        for tower_id in range(self._bst_tower_num):
            tower_fea = self._bst_tower_features[tower_id]
            tower = self._model_config.bst_towers[tower_id]

            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
                bst_layer = BSTLayer()
                tower_fea = bst_layer.bst(
                    "%s_bst" % tower.input_group,
                    tower_fea,
                    seq_size=tower.bst_config.seq_size,
                    multi_head_self_att_config=tower.bst_config.multi_head_self_att_config,
                    return_target=tower.bst_config.return_target,
                )
                tower_fea_arr.append(tower_fea)

        for tower_id in range(self._dien_tower_num):
            tower_fea = self._dien_tower_features[tower_id]
            tower = self._model_config.dien_towers[tower_id]

            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
                dien_layer = DIENLayer()
                tower_fea = dien_layer.dien(
                    name="%s_dien" % tower.input_group,
                    deep_fea=tower_fea,
                    seq_size=tower.dien_config.seq_size,
                    combine_mechanism=tower.dien_config.combine_mechanism,
                    return_target=tower.dien_config.return_target,
                )
                tower_fea_arr.append(tower_fea)

        for tower_id in range(self._can_tower_num):
            tower_fea = self._can_tower_features[tower_id]
            tower = self._model_config.can_towers[tower_id]

            with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
                can_layer = CANLayer()
                tower_fea = can_layer.call(
                    name="%s_can" % tower.input_group,
                    deep_fea=tower_fea,
                    item_vocab_size=tower.can_config.item_vocab_size,
                    mlp_units=tower.can_config.mlp_units,
                )
                tower_fea_arr.append(tower_fea)

        return tower_fea_arr

    def build_predict_graph(self):
        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, tower_fea_arr.length:%s" % (filename, str(len(tower_fea_arr))))

        all_fea = tf.concat(tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        if self._model_config.star_model_config:
            star_model_config = self._model_config.star_model_config
            domain_id = self.get_id_feature(
                star_model_config.domain_input_group, star_model_config.domain_id_col,
                use_raw_id=True
            )
            all_fea = StarTopologyFCNLayer().call(
                name="star_fcn", deep_fea=all_fea, domain_id=domain_id,
                domain_size=self._model_config.star_model_config.domain_size,
                mlp_units=self._model_config.final_dnn.hidden_units,
            )
        else:
            all_fea = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)(all_fea)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        if self._model_config.wide_towers:
            wide_fea = tf.concat(self._wide_tower_features, axis=1)
            all_fea = tf.concat([all_fea, wide_fea], axis=1)
            logging.info("%s build_predict_graph, with wide tower, all_fea.shape:%s" % (filename, str(all_fea.shape)))

        if self._model_config.bias_towers:
            for tower in self._model_config.bias_towers:
                bias_fea = self.build_input_layer(tower.input_group)
                bias_fea = tf.layers.dense(bias_fea, all_fea.shape()[-1],
                                           name="bias_tower_dense_" + tower.input_group)
                if "multiply" == tower.fusion_mode:
                    all_fea = tf.multiply(all_fea, bias_fea)
                else:
                    all_fea = tf.add(all_fea, bias_fea)

        if self._model_config.star_model_config:
            star_model_config = self._model_config.star_model_config
            logits = STARAuxiliaryNetworkLayer().call(
                name="star_aux", deep_fea=all_fea,
                domain_fea=self.get_id_feature(
                    star_model_config.domain_input_group, star_model_config.domain_id_col,
                    use_raw_id=False
                ),
                mlp_units=star_model_config.auxiliary_network_mlp_units,
            )
        else:
            logits = tf.layers.dense(all_fea, 1, name="logits")

        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
