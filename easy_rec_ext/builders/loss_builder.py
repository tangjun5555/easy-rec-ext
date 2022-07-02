# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/11/15 5:16 下午
# desc:

import os
import logging
from enum import Enum, unique
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


@unique
class LossType(Enum):
    CROSS_ENTROPY_LOSS = 0
    SOFTMAX_CROSS_ENTROPY_LOSS = 1
    L2_LOSS = 2
    PAIRWISE_LOSS = 3

    @staticmethod
    def handle(value):
        if "CROSS_ENTROPY_LOSS" == value:
            return LossType.CROSS_ENTROPY_LOSS
        elif "SOFTMAX_CROSS_ENTROPY_LOSS" == value:
            return LossType.SOFTMAX_CROSS_ENTROPY_LOSS
        elif "L2_LOSS" == value:
            return LossType.L2_LOSS
        elif "PAIRWISE_LOSS" == value:
            return LossType.PAIRWISE_LOSS
        else:
            raise NotImplemented


def build_pairwise_loss(labels, logits):
    """
    Build pair-wise loss.
    Args:
        labels: [batch_size]
        logits: [batch_size]

    Returns:
        loss: scalar
    """
    pairwise_logits = tf.expand_dims(logits, -1) - tf.expand_dims(logits, 0)
    logging.info("[pairwise_loss] pairwise logits: {}".format(pairwise_logits))

    pairwise_mask = tf.greater(
        tf.expand_dims(labels, -1) - tf.expand_dims(labels, 0), 0)
    logging.info("[pairwise_loss] mask: {}".format(pairwise_mask))

    pairwise_logits = tf.boolean_mask(pairwise_logits, pairwise_mask)
    logging.info("[pairwise_loss] after masking: {}".format(pairwise_logits))

    pairwise_pseudo_labels = tf.ones_like(pairwise_logits)
    loss = tf.losses.sigmoid_cross_entropy(pairwise_pseudo_labels, pairwise_logits)
    # set rank loss to zero if a batch has no positive sample.
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return loss


class KnowledgeDistillation(object):
    def __init__(self, pred_name: str, label_name: str,
                 pred_is_logits: bool = True, label_is_logits: bool = True,
                 loss_type: LossType = LossType.L2_LOSS,
                 loss_weight: float = 1.0, temperature: float = 1.0,
                 ):
        self.pred_name = pred_name
        self.label_name = label_name
        self.pred_is_logits = pred_is_logits
        self.label_is_logits = label_is_logits
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.temperature = temperature

    @staticmethod
    def handle(data):
        res = KnowledgeDistillation(
            data["pred_name"], data["label_name"]
        )
        if "pred_is_logits" in data:
            res.pred_is_logits = data["pred_is_logits"]
        if "label_is_logits" in data:
            res.label_is_logits = data["label_is_logits"]
        if "loss_type" in data:
            res.loss_type = LossType.handle(data["loss_type"])
        if "loss_weight" in data:
            res.loss_weight = data["loss_weight"]
        if "temperature" in data:
            res.temperature = data["temperature"]
        return res


def build_kd_loss(kds, prediction_dict, label_dict):
    """
    Build knowledge distillation loss.
    Args:
      kds: list of knowledge distillation object of type KD.
      prediction_dict: dict of predict_name to predict tensors.
      label_dict: ordered dict of label_name to label tensors.
    Return:
      knowledge distillation loss will be add to loss_dict with key: kd_loss.
    """
    loss_dict = {}
    for kd in kds:
        assert kd.pred_name in prediction_dict, \
            "invalid predict_name: %s available ones: %s" % (
                kd.pred_name, ",".join(prediction_dict.keys()))

        assert kd.pred_is_logits, "predict_name:%s must be logits" % kd.pred_name

        loss_name = "kd_loss"
        loss_name += '_' + kd.pred_name.replace('/', '_')
        loss_name += '_' + kd.label_name.replace('/', '_')

        label = label_dict[kd.label_name]
        pred = prediction_dict[kd.pred_name]

        if kd.temperature > 0 and kd.temperature != 1.0 and kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
            if kd.pred_is_logits:
                pred = pred / kd.temperature
            if kd.label_is_logits:
                label = label / kd.temperature

        if kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
            num_class = 1 if len(pred.get_shape()) < 2 else pred.get_shape()[-1]
            if num_class > 1:
                if kd.pred_is_logits:
                    pred = tf.nn.softmax(pred)
                # if kd.label_is_logits:
                #     label = tf.nn.softmax(label)
            else:
                if kd.pred_is_logits:
                    pred = tf.nn.sigmoid(pred)
                # if kd.label_is_logits:
                #     label = tf.nn.sigmoid(label)

        if kd.loss_type == LossType.CROSS_ENTROPY_LOSS:
            loss_dict[loss_name] = tf.losses.log_loss(
                label, pred, weights=kd.loss_weight
            )
        elif kd.loss_type == LossType.L2_LOSS:
            loss_dict[loss_name] = tf.losses.mean_squared_error(
                labels=label, predictions=pred, weights=kd.loss_weight,
            )
        else:
            assert False, "unsupported loss type for kd: %s" % LossType.Name(
                kd.loss_type)
    return loss_dict


class InBatchNegSoftmaxLossConfig(object):
    def __init__(self, neg_num=10, temperature=0.1,
                 ):
        self.neg_num = neg_num
        self.temperature = temperature

    @staticmethod
    def handle(data):
        res = InBatchNegSoftmaxLossConfig
        if "neg_num" in data:
            res.neg_num = data["neg_num"]
        if "temperature" in data:
            res.temperature = data["temperature"]
        return res


def build_inbatch_neg_softmax_loss(query_encoder, doc_encoder, loss_config):
    """
    Args:
        query_encoder: (batch_size, dim)
        doc_encoder: (batch_size, dim)
        loss_config: InBatchNegSoftmaxLossConfig
    Returns:
        loss
    """
    doc_encoder_fd = doc_encoder
    for i in range(loss_config.neg_num):
        ss = tf.gather(doc_encoder, tf.random.shuffle(tf.range(tf.shape(doc_encoder)[0]), seed=555))
        doc_encoder_fd = tf.concat([doc_encoder_fd, ss], axis=0)
    query_encoder_fd = tf.tile(query_encoder, [loss_config.neg_num + 1, 1])
    logging.info("%s build_inbatch_neg_softmax_loss, query_encoder_fd.shape:%s, doc_encoder_fd.shape:%s" %
                 (filename, str(tf.shape(query_encoder_fd)), str(tf.shape(doc_encoder_fd)))
                 )

    similarity = tf.reduce_sum(
        tf.multiply(query_encoder_fd, doc_encoder_fd),
        axis=1,
        keepdims=True,
    )
    similarity = tf.transpose(tf.reshape(tf.transpose(similarity), [loss_config.neg_num + 1, -1]))
    logging.info("%s build_inbatch_neg_softmax_loss, similarity.shape:%s" %
                 (filename, str(tf.shape(similarity)))
                 )

    prob = tf.exp(similarity / loss_config.temperature) / tf.reduce_sum(tf.exp(similarity / loss_config.temperature),
                                                                        axis=1, keepdims=True)
    pos_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_mean(tf.math.log(pos_prob))
    return loss
