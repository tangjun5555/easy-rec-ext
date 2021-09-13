# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 12:22 下午
# desc:

import tensorflow_recommenders_addons as tfra

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


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

    optimizer = None
    summary_vars = []

    if optimizer_config.optimizer_type == "sgd_optimizer":
        config = optimizer_config.sgd_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    if optimizer_config.optimizer_type == "adagrad_optimizer":
        config = optimizer_config.adagrad_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.AdagradOptimizer(learning_rate)

    if optimizer_config.optimizer_type == "adam_optimizer":
        config = optimizer_config.adam_optimizer
        learning_rate = _create_learning_rate(config.learning_rate)
        summary_vars.append(learning_rate)
        optimizer = tf.train.AdamOptimizer(
            learning_rate, beta1=config.beta1, beta2=config.beta2
        )

    if optimizer is None:
        raise ValueError("Optimizer %s not supported." % optimizer_config.optimizer_type)

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
