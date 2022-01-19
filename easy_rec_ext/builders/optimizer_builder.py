# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 12:22 下午
# desc:

import os
import logging
import tensorflow as tf
import tensorflow_recommenders_addons as tfra

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
filename = str(os.path.basename(__file__)).split(".")[0]


class ConstantLearningRate(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    @staticmethod
    def handle(data):
        res = ConstantLearningRate()
        if "learning_rate" in data:
            res.learning_rate = data["learning_rate"]
        return res

    def __str__(self):
        return str(self.__dict__)


class ExponentialDecayLearningRate(object):
    def __init__(self, initial_learning_rate=0.01, decay_steps=20000, decay_factor=0.95, min_learning_rate=0.0001):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate

    @staticmethod
    def handle(data):
        res = ExponentialDecayLearningRate()
        if "initial_learning_rate" in data:
            res.initial_learning_rate = data["initial_learning_rate"]
        if "decay_steps" in data:
            res.decay_steps = data["decay_steps"]
        if "decay_factor" in data:
            res.decay_factor = data["decay_factor"]
        if "min_learning_rate" in data:
            res.min_learning_rate = data["min_learning_rate"]
        return res

    def __str__(self):
        return str(self.__dict__)


class LearningRate(object):
    def __init__(self, learning_rate_type="exponential_decay_learning_rate",
                 constant_learning_rate=ConstantLearningRate(),
                 exponential_decay_learning_rate=ExponentialDecayLearningRate(),
                 ):
        self.learning_rate_type = learning_rate_type
        self.constant_learning_rate = constant_learning_rate
        self.exponential_decay_learning_rate = exponential_decay_learning_rate

    @staticmethod
    def handle(data):
        res = LearningRate()
        if "learning_rate_type" in data:
            res.learning_rate_type = data["learning_rate_type"]
        if "constant_learning_rate" in data:
            res.constant_learning_rate = ConstantLearningRate.handle(data["constant_learning_rate"])
        if "exponential_decay_learning_rate" in data:
            res.exponential_decay_learning_rate = ExponentialDecayLearningRate.handle(
                data["exponential_decay_learning_rate"]
            )
        return res

    def __str__(self):
        return str(self.__dict__)


class SgdOptimizer(object):
    def __init__(self, learning_rate=LearningRate()):
        self.learning_rate = learning_rate

    @staticmethod
    def handle(data):
        res = SgdOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        return res

    def __str__(self):
        return str(self.__dict__)


class MomentumOptimizer(object):
    def __init__(self, learning_rate=LearningRate(), momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    @staticmethod
    def handle(data):
        res = AdagradOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        if "momentum" in data:
            res.momentum = data["momentum"]
        return res

    def __str__(self):
        return str(self.__dict__)


class AdagradOptimizer(object):
    def __init__(self, learning_rate=LearningRate(), initial_accumulator_value=0.1):
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value

    @staticmethod
    def handle(data):
        res = AdagradOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        if "initial_accumulator_value" in data:
            res.initial_accumulator_value = data["initial_accumulator_value"]
        return res

    def __str__(self):
        return str(self.__dict__)


class AdamOptimizer(object):
    def __init__(self, learning_rate=LearningRate(), beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @staticmethod
    def handle(data):
        res = AdamOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        if "beta1" in data:
            res.beta1 = data["beta1"]
        if "beta2" in data:
            res.beta2 = data["beta2"]
        if "epsilon" in data:
            res.epsilon = data["epsilon"]
        return res

    def __str__(self):
        return str(self.__dict__)


class Optimizer(object):
    def __init__(self, optimizer_type="sgd_optimizer",
                 sgd_optimizer=SgdOptimizer(),
                 momentum_optimizer=MomentumOptimizer(),
                 adagrad_optimizer=AdagradOptimizer(),
                 adam_optimizer=AdamOptimizer()
                 ):
        self.optimizer_type = optimizer_type
        self.sgd_optimizer = sgd_optimizer
        self.momentum_optimizer = momentum_optimizer
        self.adagrad_optimizer = adagrad_optimizer
        self.adam_optimizer = adam_optimizer

    @staticmethod
    def handle(data):
        res = Optimizer()
        if "optimizer_type" in data:
            res.optimizer_type = data["optimizer_type"]
        if "sgd_optimizer" in data:
            res.sgd_optimizer = SgdOptimizer.handle(data["sgd_optimizer"])
        if "momentum_optimizer" in data:
            res.momentum_optimizer = MomentumOptimizer.handle(data["momentum_optimizer"])
        if "adagrad_optimizer" in data:
            res.adagrad_optimizer = AdagradOptimizer.handle(data["adagrad_optimizer"])
        if "adam_optimizer" in data:
            res.adam_optimizer = AdamOptimizer.handle(data["adam_optimizer"])
        return res

    def __str__(self):
        return str(self.__dict__)


def build(optimizer_config):
    """
    Create optimizer based on config.

    Args:
      optimizer_config:

    Returns:
      An optimizer and a list of variables for summary.

    Raises:
      ValueError: when using an unsupported input data type.
    """

    summary_vars = []
    optimizer = None
    config = None

    if optimizer_config.optimizer_type == "sgd_optimizer":
        config = optimizer_config.sgd_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    if optimizer_config.optimizer_type == "momentum_optimizer":
        config = optimizer_config.momentum_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=config.momentum)

    if optimizer_config.optimizer_type == "adagrad_optimizer":
        config = optimizer_config.adagrad_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=config.initial_accumulator_value)

    if optimizer_config.optimizer_type == "adam_optimizer":
        config = optimizer_config.adam_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.AdamOptimizer(
            learning_rate, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon
        )

    if optimizer is None:
        raise ValueError("Optimizer %s not supported." % optimizer_config.optimizer_type)
    else:
        logging.info("%s use %s[%s]" % (filename, optimizer_config.optimizer_type, str(config)))

    optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
    return optimizer, summary_vars


def exponential_decay_with_burnin(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  min_learning_rate=0.0001,
                                  burnin_learning_rate=0.0,
                                  burnin_steps=0,
                                  staircase=True):
    """
    Exponential decay schedule with burn-in period.

    In this schedule, learning rate is fixed at burnin_learning_rate
    for a fixed period, before transitioning to a regular exponential
    decay schedule.

    Args:
      global_step: int tensor representing global step.
      learning_rate_base: base learning rate.
      learning_rate_decay_steps: steps to take between decaying the learning rate.
        Note that this includes the number of burn-in steps.
      learning_rate_decay_factor: multiplicative factor by which to decay
        learning rate.
      burnin_learning_rate: initial learning rate during burn-in period.  If
        0.0 (which is the default), then the burn-in learning rate is simply
        set to learning_rate_base.
      burnin_steps: number of steps to use burnin learning rate.
      min_learning_rate: the minimum learning rate.
      staircase: whether use staircase decay.

    Returns:
      a (scalar) float tensor representing learning rate
    """
    if burnin_learning_rate == 0:
        burnin_rate = learning_rate_base
    else:
        slope = (learning_rate_base - burnin_learning_rate) / burnin_steps
        burnin_rate = slope * tf.cast(global_step, tf.float32) + burnin_learning_rate
    post_burnin_learning_rate = tf.train.exponential_decay(
        learning_rate_base,
        global_step - burnin_steps,
        learning_rate_decay_steps,
        learning_rate_decay_factor,
        staircase=staircase
    )
    return tf.maximum(
        tf.where(
            tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)),
            burnin_rate,
            post_burnin_learning_rate
        ),
        min_learning_rate,
        name="learning_rate"
    )


def _create_learning_rate(learning_rate_config):
    """
    Create optimizer learning rate based on config.

    Args:
      learning_rate_config:

    Returns:
      A learning rate.

    Raises:
      ValueError: when using an unsupported input data type.
    """
    learning_rate = None

    if learning_rate_config.learning_rate_type == "constant_learning_rate":
        config = learning_rate_config.constant_learning_rate
        learning_rate = tf.constant(config.learning_rate, dtype=tf.float32, name="learning_rate")

    if learning_rate_config.learning_rate_type == "exponential_decay_learning_rate":
        config = learning_rate_config.exponential_decay_learning_rate
        learning_rate = exponential_decay_with_burnin(
            tf.train.get_or_create_global_step(),
            config.initial_learning_rate,
            config.decay_steps,
            config.decay_factor,
            min_learning_rate=config.min_learning_rate,
        )

    if learning_rate is None:
        raise ValueError("Learning_rate %s not supported." % learning_rate_config.learning_rate_type)

    return learning_rate
