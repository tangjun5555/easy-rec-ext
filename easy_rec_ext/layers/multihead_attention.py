# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/3/21 5:41 PM
# desc:

import os
import logging
import numpy as np
import tensorflow as tf
from easy_rec_ext.core import embedding_ops

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class MultiHeadAttention(object):
    def __init__(self, name, head_num, head_size, feature_num, l2_reg=None, use_res=False):
        """
        Initializes a `MultiHeadAttention` Layer.
        Args:
          name: scope of the MultiHeadAttention, so that the parameters could be separated from other MultiHeadAttention
          head_num: The number of heads
          head_size: The dimension of a head
          feature_num: The number of Feature
          l2_reg: l2 regularizer
          use_res: Whether to use residual connections before output.
        """
        self._name = name
        self._head_num = head_num
        self._head_size = head_size
        self._feature_num = feature_num
        self._l2_reg = l2_reg
        self._use_res = use_res

        self.positional_encoding_type = None
        self.hist_mask = None

    def _split_multihead_qkv(self, q, k, v):
        """
        Split multiple heads.
        Args:
          q: Query matrix of shape [bs, feature_num, head_num * head_size].
          k: Key matrix of shape [bs, feature_num, head_num * head_size].
          v: Value matrix of shape [bs, feature_num, head_num * head_size].
        Returns:
          q: Query matrix of shape [bs, head_num, feature_num, head_size].
          k: Key matrix of shape [bs, head_num, feature_num, head_size].
          v: Value matrix of shape [bs, head_num, feature_num, head_size].
        """
        reshaped_q = tf.reshape(
            q, shape=[-1, self._feature_num, self._head_num, self._head_size]
        )
        q = tf.transpose(reshaped_q, perm=[0, 2, 1, 3])
        reshaped_k = tf.reshape(
            k, shape=[-1, self._feature_num, self._head_num, self._head_size]
        )
        k = tf.transpose(reshaped_k, perm=[0, 2, 1, 3])
        reshaped_v = tf.reshape(
            v, shape=[-1, self._feature_num, self._head_num, self._head_size]
        )
        v = tf.transpose(reshaped_v, perm=[0, 2, 1, 3])
        return q, k, v

    def _scaled_dot_product_attention(self, q, k, v):
        """
        Calculate scaled dot product attention by q, k and v.
        Args:
          q: Query matrix of shape [bs, head_num, feature_num, head_size].
          k: Key matrix of shape [bs, head_num, feature_num, head_size].
          v: Value matrix of shape [bs, head_num, feature_num, head_size].
        Returns:
          Value matrix of shape [bs, head_num, feature_num, head_size].
        """
        product = tf.linalg.matmul(a=q, b=k, transpose_b=True) / (self._head_size ** -0.5)
        logging.info(
            "%s _scaled_dot_product_attention, %s, product.shape:%s" % (filename, self._name, str(product.shape)))

        if self.hist_mask is not None:
            mask = self.hist_mask
        else:
            mask = tf.math.greater_equal(tf.math.abs(product), 1e-9)
        padding = tf.ones_like(product) * (-2 ** 32 + 1)
        product = tf.where(mask, product, padding)
        weights = tf.nn.softmax(product)
        out = tf.linalg.matmul(weights, v)
        return out

    def _compute_qkv(self, q, k, v):
        """
        Calculate q, k and v matrices.
        Args:
          q: Query matrix of shape [bs, feature_num, d_model].
          k: Key matrix of shape [bs, feature_num, d_model].
          v: Value matrix of shape [bs, feature_num, d_model].
        Returns:
          q: Query matrix of shape [bs, feature_num, head_size * n_head].
          k: Key matrix of shape [bs, feature_num, head_size * n_head].
          v: Value matrix of shape [bs, feature_num, head_size * n_head].
        """
        q = tf.layers.dense(
            q,
            self._head_num * self._head_size,
            use_bias=False,
            kernel_regularizer=self._l2_reg,
            name="%s/%s/dnn" % (self._name, "query")
        )
        k = tf.layers.dense(
            k,
            self._head_num * self._head_size,
            use_bias=False,
            kernel_regularizer=self._l2_reg,
            name="%s/%s/dnn" % (self._name, "key")
        )
        v = tf.layers.dense(
            v,
            self._head_num * self._head_size,
            use_bias=False,
            kernel_regularizer=self._l2_reg,
            name="%s/%s/dnn" % (self._name, "value")
        )

        if "sinusoidal" == self.positional_encoding_type:
            T = self._feature_num
            num_units = self._head_num * self._head_size
            position_enc = np.array([
                [pos / np.power(10000, 2. * (i // 2) / num_units) for i in range(num_units)]
                for pos in range(T)])
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_ind = tf.expand_dims(tf.range(T), 0)
            q = q + tf.nn.embedding_lookup(tf.identity(position_enc), position_ind)
            k = k + tf.nn.embedding_lookup(tf.identity(position_enc), position_ind)
            v = v + tf.nn.embedding_lookup(tf.identity(position_enc), position_ind)
        elif "learned" == self.positional_encoding_type:
            T = self._feature_num
            num_units = self._head_num * self._head_size
            position_enc = embedding_ops.get_embedding_variable(
                name=self._name + "_position_enc",
                dim=num_units,
                vocab_size=T,
            )
            position_ind = tf.expand_dims(tf.range(T), 0)
            tf.identity()
            q = q + tf.nn.embedding_lookup(position_enc, position_ind)
            k = k + tf.nn.embedding_lookup(position_enc, position_ind)
            v = v + tf.nn.embedding_lookup(position_enc, position_ind)
        else:
            logging.info("%s %s don't use positional_encoding" % (filename, self._name))

        return q, k, v

    def _combine_heads(self, multi_head_tensor):
        """
        Combine the results of multiple heads.
        Args:
          multi_head_tensor: Result matrix of shape [bs, head_num, feature_num, head_size].
        Returns:
          out: Result matrix of shape [bs, feature_num, head_num * head_size].
        """
        x = tf.transpose(multi_head_tensor, perm=[0, 2, 1, 3])
        out = tf.reshape(x, shape=[-1, x.shape[1], x.shape[2] * x.shape[3]])
        return out

    def __call__(self, attention_input):
        """
        Build multiple heads attention layer.
        Args:
          attention_input: The input of interacting layer, has a shape of [bs, feature_num, d_model].
        Returns:
          out: The output of multi head attention layer, has a shape of [bs, feature_num, head_num * head_size].
        """
        assert isinstance(attention_input, list) and len(attention_input) == 3
        ori_q = attention_input[0]
        ori_k = attention_input[1]
        ori_v = attention_input[2]

        q, k, v = self._compute_qkv(ori_q, ori_k, ori_v)
        q, k, v = self._split_multihead_qkv(q, k, v)
        multi_head_tensor = self._scaled_dot_product_attention(q, k, v)
        out = self._combine_heads(multi_head_tensor)

        if self._use_res:
            W_0_x = tf.layers.dense(
                ori_v,
                out.shape[2],
                use_bias=False,
                kernel_regularizer=self._l2_reg,
                name="%s/res/dnn" % (self._name)
            )
            res_out = tf.nn.relu(out + W_0_x)
            return res_out
        else:
            return out


class MultiHeadSelfAttentionConfig(object):
    def __init__(self, head_num, head_size, feature_num, use_res=False, positional_encoding_type: str = None):
        self.head_num = head_num
        self.head_size = head_size
        self.feature_num = feature_num
        self.use_res = use_res
        self.positional_encoding_type = positional_encoding_type

    @staticmethod
    def handle(data):
        res = MultiHeadSelfAttentionConfig(data["head_num"], data["head_size"], data["feature_num"])
        if "use_res" in data:
            res.use_res = data["use_res"]
        if "positional_encoding_type" in data:
            res.positional_encoding_type = data["positional_encoding_type"]
        return res


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, name, head_num, head_size, feature_num,
                 l2_reg=None, use_res=False, positional_encoding_type: str = None):
        super(MultiHeadSelfAttention, self).__init__(name, head_num, head_size, feature_num, l2_reg, use_res)
        self.positional_encoding_type = positional_encoding_type

    def __call__(self, deep_fea, cur_seq_len=None):
        """
        Args:
            deep_fea: input, [bs, feature_num, d_model].
            cur_seq_len: [bs]
        Returns:
            output: [bs, feature_num, head_num * head_size].
        """

        if cur_seq_len is not None:
            hist_mask = tf.sequence_mask(
                cur_seq_len, maxlen=self._feature_num)  # [B, seq_size]
            hist_mask = tf.reshape(tf.tile(hist_mask, [1, self._feature_num]),
                                   (-1, self._feature_num, self._feature_num))
            self.hist_mask = tf.reshape(tf.tile(hist_mask, [1, self._head_num, 1]),
                                        (-1, self._head_num, self._feature_num, self._feature_num))
        return super(MultiHeadSelfAttention, self).__call__([deep_fea, deep_fea, deep_fea])
