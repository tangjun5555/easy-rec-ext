# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 10:59 下午
# desc: Behavior Sequence Transformer

import os
import logging
import tensorflow as tf
from easy_rec_ext.core import regularizers
from easy_rec_ext.layers import dnn, layer_norm
from easy_rec_ext.layers.multihead_attention import MultiHeadSelfAttention, MultiHeadSelfAttentionConfig
from easy_rec_ext.model.rank_model import RankModel

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class BSTConfig(object):
    def __init__(self, seq_size,
                 multi_head_self_att_config: MultiHeadSelfAttentionConfig,
                 return_target: bool = True,
                 ):
        self.seq_size = seq_size
        self.multi_head_self_att_config = multi_head_self_att_config
        self.return_target = return_target

    @staticmethod
    def handle(data):
        res = BSTConfig(data["seq_size"], MultiHeadSelfAttentionConfig.handle(data["multi_head_self_att_config"]))
        if "return_target" in data:
            res.return_target = data["return_target"]
        return res

    def __str__(self):
        return str(self.__dict__)


class BSTTower(object):
    def __init__(self, input_group, bst_config: BSTConfig):
        self.input_group = input_group
        self.bst_config = bst_config

    @staticmethod
    def handle(data):
        bst_config = BSTConfig.handle(data["bst_config"])
        res = BSTTower(data["input_group"], bst_config)
        return res

    def __str__(self):
        return str(self.__dict__)


class BSTLayer(object):
    def dnn_net(self, net, dnn_units, name):
        with tf.variable_scope(name_or_scope=name, reuse=tf.AUTO_REUSE):
            for idx, units in enumerate(dnn_units):
                net = tf.layers.dense(
                    net, units=units, activation=tf.nn.relu, name="%s_%d" % (name, idx)
                )
        return net

    def add_and_norm(self, net_1, net_2, emb_dim, name):
        net = tf.add(net_1, net_2)
        layer = layer_norm.LayerNormalization(emb_dim, name)
        net = layer(net)
        return net

    def bst(self, name, deep_fea, seq_size, multi_head_self_att_config, return_target=True):
        cur_id, hist_id_col, seq_len = deep_fea["key"], deep_fea["hist_seq_emb"], deep_fea["hist_seq_len"]
        emb_dim = hist_id_col.get_shape().as_list()[2]
        all_ids = tf.concat([tf.expand_dims(cur_id, 1), hist_id_col], axis=1)  # b, seq_size + 1, emb_dim

        attention_net = MultiHeadSelfAttention(
                name=name + "_" + "MultiHeadSelfAttention",
                head_num=multi_head_self_att_config.head_num,
                head_size=multi_head_self_att_config.head_size,
                feature_num=1 + seq_size,
                l2_reg=None,
                use_res=multi_head_self_att_config.use_res,
        )(all_ids, 1 + seq_len)
        logging.info("%s %s, attention_net.shape:%s" % (filename, name, str(attention_net.shape)))
        attention_net = self.dnn_net(attention_net, [emb_dim], name + "_" + "attention_net_dnn")
        logging.info("%s %s, attention_net.shape:%s" % (filename, name, str(attention_net.shape)))

        tmp_net = self.add_and_norm(all_ids, attention_net, emb_dim, name=name + "_" + "add_and_norm_1")
        feed_forward_net = self.dnn_net(tmp_net, [emb_dim], name + "_" + "feed_forward_net")
        net = self.add_and_norm(tmp_net, feed_forward_net, emb_dim, name=name + "_" + "add_and_norm_2")

        bst_output = tf.reshape(net, [-1, (1 + seq_size) * emb_dim])
        if return_target:
            bst_output = tf.concat([bst_output, cur_id], axis=1)
        logging.info("bst %s, bst_output.shape:%s" % (name, str(bst_output.shape)))
        return bst_output


class BST(RankModel, BSTLayer):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(BST, self).__init__(model_config, feature_config, features, labels, is_training)

        self._dnn_tower_num = len(self._model_config.dnn_towers) if self._model_config.dnn_towers else 0
        self._dnn_tower_features = []
        for tower_id in range(self._dnn_tower_num):
            tower = self._model_config.dnn_towers[tower_id]
            tower_feature = self.build_input_layer(tower.input_group)
            self._dnn_tower_features.append(tower_feature)

        self._bst_tower_num = len(self._model_config.bst_towers) if self._model_config.bst_towers else 0
        self._bst_tower_features = []
        for tower_id in range(self._bst_tower_num):
            tower = self._model_config.bst_towers[tower_id]
            tower_feature = self.build_seq_att_input_layer(tower.input_group)
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["key"]])
            regularizers.apply_regularization(self._emb_reg, weights_list=[tower_feature["hist_seq_emb"]])
            self._bst_tower_features.append(tower_feature)

        logging.info("all tower num: {0}".format(self._dnn_tower_num + self._bst_tower_num))
        logging.info("dnn tower num: {0}".format(self._dnn_tower_num))
        logging.info("bst tower num: {0}".format(self._bst_tower_num))

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
            tower_dnn = dnn.DNN(
                tower.dnn_config,
                self._l2_reg,
                "%s_dnn" % tower_name,
                self._is_training,
            )
            tower_fea = tower_dnn(tower_fea)
            tower_fea_arr.append(tower_fea)

        for tower_id in range(self._bst_tower_num):
            tower_fea = self._bst_tower_features[tower_id]
            tower = self._model_config.bst_towers[tower_id]
            tower_fea = self.bst(
                "%s_bst" % tower.input_group,
                tower_fea,
                seq_size=tower.bst_config.seq_size,
                multi_head_self_att_config=tower.bst_config.multi_head_self_att_config,
                return_target=tower.bst_config.return_target,
            )
            tower_fea_arr.append(tower_fea)

        all_fea = tf.concat(tower_fea_arr, axis=1)
        final_dnn = dnn.DNN(self._model_config.final_dnn, self._l2_reg, "final_dnn", self._is_training)
        all_fea = final_dnn(all_fea)
        logging.info("BST build_predict_graph, logits.shape:%s" % (str(all_fea.shape)))

        logits = tf.layers.dense(all_fea, 1, name="logits")
        logits = tf.reshape(logits, (-1,))
        probs = tf.sigmoid(logits, name="probs")

        prediction_dict = dict()
        prediction_dict["probs"] = probs
        self._add_to_prediction_dict(prediction_dict)
        return self._prediction_dict
