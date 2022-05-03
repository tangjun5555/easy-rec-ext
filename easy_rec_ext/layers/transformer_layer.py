# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/11/9 11:20 下午
# desc:

import os
import logging
import tensorflow as tf
from easy_rec_ext.layers import layer_norm, multihead_attention

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class TransformerEncodeLayer(object):
    def __int__(self, name, multi_head_self_att_config: multihead_attention.MultiHeadSelfAttentionConfig):
        self.name = name
        self.multi_head_self_att_config = multi_head_self_att_config

    def add_and_norm(self, net_1, net_2, emb_dim):
        net = tf.add(net_1, net_2)
        layer = layer_norm.LayerNormalization(emb_dim, self.name)
        net = layer(net)
        return net

    def feed_forward_net(self, net, emb_dim):
        net = tf.layers.dense(
            net, units=emb_dim, activation=tf.nn.relu, name="%s_ffn" % self.name
        )
        return net

    def __call__(self, deep_fea, cur_seq_len=None):
        """
        Args:
            deep_fea: input, [bs, feature_num, d_model].
            cur_seq_len: [bs]
        Returns:
            output: [bs, feature_num, head_num * head_size].
        """
        logging.info("%s %s, deep_fea.shape:%s" % (filename, self.name, str(deep_fea.shape)))
        output_size = self.multi_head_self_att_config.head_num * self.multi_head_self_att_config.head_size

        attention_net = multihead_attention.MultiHeadSelfAttention(
            name=self.name + "_" + "MultiHeadSelfAttention",
            head_num=self.multi_head_self_att_config.head_num,
            head_size=self.multi_head_self_att_config.head_size,
            feature_num=self.multi_head_self_att_config.feature_num,
            l2_reg=None,
            use_res=self.multi_head_self_att_config.use_res,
            positional_encoding_type=self.multi_head_self_att_config.positional_encoding_type,
        )(deep_fea)
        logging.info("%s %s, attention_net.shape:%s" % (filename, self.name, str(attention_net.shape)))

        deep_fea = tf.layers.dense(
            deep_fea,
            output_size,
            use_bias=False,
            name="%s_projection" % self.name
        )
        logging.info("%s %s, deep_fea.shape:%s" % (filename, self.name, str(deep_fea.shape)))

        tmp_net = self.add_and_norm(deep_fea, attention_net, output_size)
        feed_forward_net = self.feed_forward_net(tmp_net, output_size)
        output = self.add_and_norm(tmp_net, feed_forward_net, output_size)
        return output
