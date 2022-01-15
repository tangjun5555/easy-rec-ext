# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/1/3 10:13 AM
# desc: Star Topology Adaptive Recommender


import os
import logging
import tensorflow as tf
from easy_rec_ext.core import embedding_ops
from easy_rec_ext.utils import variable_util

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class STARModelConfig(object):
    def __init__(self, use_star_topology_fcn_layer: bool = True,
                 use_auxiliary_network_layer: bool = True):
        self.use_star_topology_fcn_layer = use_star_topology_fcn_layer
        self.use_auxiliary_network_layer = use_auxiliary_network_layer

    @staticmethod
    def handle(data):
        res = STARModelConfig()
        if "use_star_topology_fcn_layer" in data:
            res.use_star_topology_fcn_layer = data["use_star_topology_fcn_layer"]
        if "use_auxiliary_network_layer" in data:
            res.use_auxiliary_network_layer = data["use_auxiliary_network_layer"]
        return res

    def __str__(self):
        return str(self.__dict__)


class StarTopologyFCNLayer(object):
    def call(self, name, deep_fea, domain_id, domain_size, mlp_units):
        input_dimension = deep_fea.get_shape().as_list()[-1]

        domain_weight_dim = 0
        weight_w_dim = []
        weight_b_dim = []
        x = input_dimension
        for y in mlp_units:
            domain_weight_dim += x * y
            domain_weight_dim += 1
            weight_w_dim.append([x, y])
            weight_b_dim.append(1)
            x = y

        logging.info("%s call, input_dimension:%d, mlp_units:%s" % (filename, input_dimension, str(mlp_units)))

        domain_weight_embedding = embedding_ops.get_embedding_variable(
            name=name + "_domain",
            dim=domain_weight_dim,
            vocab_size=domain_size,
        )
        domain_weight_value = embedding_ops.safe_embedding_lookup(
            domain_weight_embedding, domain_id,
        )

        share_weight_value = variable_util.get_normal_variable(
            scope=name,
            name=name + "_share_weight",
            shape=[1, domain_weight_dim]
        )

        mlp_weight, mlp_bias = [], []
        idx = 0
        for w, b in zip(weight_w_dim, weight_b_dim):
            domain_w = tf.slice(domain_weight_value, [0, idx], [-1, w[0] * w[1]])
            share_w = tf.slice(share_weight_value, [0, idx], [-1, w[0] * w[1]])
            mlp_weight.append(tf.multiply(domain_w, share_w))
            idx += w[0] * w[1]

            domain_b = tf.slice(domain_weight_value, [0, idx], [-1, b])
            share_b = tf.slice(share_weight_value, [0, idx], [-1, b])
            mlp_bias.append(domain_b + share_b)
            idx += b

        h = deep_fea
        for j, (w, b) in enumerate(zip(mlp_weight, mlp_bias)):
            h = tf.matmul(h, w)
            h = h + b
            h = tf.nn.relu(h)

        return h


class AuxiliaryNetworkLayer(object):
    def call(self, name, deep_fea, domain_fea, mlp_units):
        raw_logit = tf.layers.dense(deep_fea, 1, name=name + "_raw_logit")

        domain_logit = domain_fea
        for i in range(len(mlp_units)):
            unit = mlp_units[i]
            domain_logit = tf.layers.dense(domain_logit, unit,
                                           activation=tf.nn.relu,
                                           name=name + "_domain_logit_" + str(i + 1),
                                           )
        domain_logit = tf.layers.dense(deep_fea, 1, name=name + "_domain_logit")

        return raw_logit + domain_logit
