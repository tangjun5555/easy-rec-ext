# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/10/20 2:52 下午
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.model.multi_tower import MultiTower


if tf.__version__ >= "2.0":
    tf = tf.compat.v1

filename = str(os.path.basename(__file__)).split(".")[0]


class MMoE(MultiTower):
    def __init__(self,
                 model_config,
                 feature_config,
                 features,
                 labels=None,
                 is_training=False,
                 ):
        super(MMoE, self).__init__(model_config, feature_config, features, labels, is_training)

    def build_predict_graph(self):
        tower_fea_arr = self.build_tower_fea_arr()
        logging.info("%s build_predict_graph, tower_fea_arr.length:%s" % (filename, str(len(tower_fea_arr))))

        all_fea = tf.concat(tower_fea_arr, axis=1)
        logging.info("%s build_predict_graph, all_fea.shape:%s" % (filename, str(all_fea.shape)))


