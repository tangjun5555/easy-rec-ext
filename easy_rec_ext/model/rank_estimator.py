# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:34 下午
# desc:

import os
import time
import logging
from collections import OrderedDict
from easy_rec_ext.model import DIN, BST, AITM
from easy_rec_ext.builders import optimizer_builder

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class RankEstimator(tf.estimator.Estimator):
    def __init__(self, pipeline_config):
        self._pipeline_config = pipeline_config
        super(RankEstimator, self).__init__(
            model_fn=self._model_fn,
            model_dir=self._pipeline_config.model_dir,
            config=None,
            params=None,
            warm_start_from=None
        )

    @property
    def _rank_model(self):
        if self._pipeline_config.model_config.model_class == "din":
            model = DIN
        elif self._pipeline_config.model_config.model_class == "bst":
            model = BST
        elif self._pipeline_config.model_config.model_class == "aitm":
            model = AITM
        else:
            raise ValueError("model_class:%s not supported." % self._pipeline_config.model_config.model_class)
        return model

    def _train_model_fn(self, features, labels):
        model = self._rank_model(
            self._pipeline_config.model_config,
            self._pipeline_config.feature_configs,
            features,
            labels,
            is_training=True
        )

        loss_dict = model.build_loss_graph()

        # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # if regularization_losses:
        #     regularization_losses = [
        #         reg_loss.get() if hasattr(reg_loss, "get") else reg_loss
        #         for reg_loss in regularization_losses
        #     ]
        #     regularization_losses = tf.add_n(regularization_losses, name="regularization_loss")
        #     loss_dict["regularization_loss"] = regularization_losses

        loss = tf.add_n(list(loss_dict.values()))
        loss_dict["total_loss"] = loss

        for key in loss_dict:
            tf.summary.scalar(key, loss_dict[key], family="loss")

        # update op, usually used for batch-norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            update_op = tf.group(*update_ops, name="update_barrier")
            with tf.control_dependencies([update_op]):
                loss = tf.identity(loss, name="total_loss")

        # build optimizer
        optimizer_config = self._pipeline_config.train_config.optimizer_config
        optimizer, learning_rate = optimizer_builder.build(optimizer_config)
        tf.summary.scalar("learning_rate", learning_rate[0])

        hooks = []

        # # for distributed and synced training
        # if self.train_config.sync_replicas and run_config.num_worker_replicas > 1:
        #     logging.info("sync_replicas: num_worker_replias = %d" %
        #                  run_config.num_worker_replicas)
        #     optimizer = tf.train.SyncReplicasOptimizer(
        #         optimizer,
        #         replicas_to_aggregate=run_config.num_worker_replicas,
        #         total_num_replicas=run_config.num_worker_replicas)
        #     hooks.append(
        #         optimizer.make_session_run_hook(run_config.is_chief, num_tokens=0))
        #
        # # add barrier for no strategy case
        # if run_config.num_worker_replicas > 1 and \
        #     self.train_config.train_distribute == DistributionStrategy.NoStrategy:
        #     hooks.append(
        #         estimator_utils.ExitBarrierHook(run_config.num_worker_replicas,
        #                                         run_config.is_chief, self.model_dir))

        # summaries = ["global_gradient_norm"]
        # if self.train_config.summary_model_vars:
        #     summaries.extend(["gradient_norm", "gradients"])
        #
        # gradient_clipping_by_norm = self.train_config.gradient_clipping_by_norm
        # if gradient_clipping_by_norm <= 0:
        #     gradient_clipping_by_norm = None
        #
        # gradient_multipliers = None
        # if self.train_config.optimizer_config.HasField(
        #     "embedding_learning_rate_multiplier"):
        #     gradient_multipliers = {
        #         var:
        #             self.train_config.optimizer_config.embedding_learning_rate_multiplier
        #         for var in tf.trainable_variables()
        #         if "embedding_weights:" in var.name or
        #            "/embedding_weights/part_" in var.name
        #     }

        # optimize loss
        # colocate_gradients_with_ops=True means to compute gradients
        # on the same device on which op is processes in forward process
        # train_op = optimizers.optimize_loss(
        #     loss=loss,
        #     global_step=tf.train.get_global_step(),
        #     learning_rate=None,
        #     clip_gradients=gradient_clipping_by_norm,
        #     optimizer=optimizer,
        #     gradient_multipliers=gradient_multipliers,
        #     variables=tf.trainable_variables(),
        #     summaries=summaries,
        #     colocate_gradients_with_ops=True,
        #     not_apply_grad_after_first_step=run_config.is_chief and
        #                                     self._pipeline_config.data_config.chief_redundant,
        #     name="")  # Preventing scope prefix on all variables.

        # # online evaluation
        # metric_update_op_dict = None
        # if self.eval_config.eval_online:
        #     metric_update_op_dict = {}
        #     metric_dict = model.build_metric_graph(self.eval_config)
        #     for k, v in metric_dict.items():
        #         metric_update_op_dict["%s/batch" % k] = v[1]
        #         tf.summary.scalar("%s/batch" % k, v[1])
        #     train_op = tf.group([train_op] + list(metric_update_op_dict.values()))
        #     if estimator_utils.is_chief():
        #         hooks.append(
        #             estimator_utils.OnlineEvaluationHook(
        #                 metric_dict=metric_dict, output_dir=self.model_dir))

        # if self.train_config.HasField("fine_tune_checkpoint"):
        #     fine_tune_ckpt = self.train_config.fine_tune_checkpoint
        #     logging.warning("will restore from %s" % fine_tune_ckpt)
        #     fine_tune_ckpt_var_map = self.train_config.fine_tune_ckpt_var_map
        #     force_restore = self.train_config.force_restore_shape_compatible
        #     restore_hook = model.restore(
        #         fine_tune_ckpt,
        #         include_global_step=False,
        #         ckpt_var_map_path=fine_tune_ckpt_var_map,
        #         force_restore_shape_compatible=force_restore)
        #     if restore_hook is not None:
        #         hooks.append(restore_hook)

        # logging
        logging_dict = OrderedDict()
        logging_dict["lr"] = learning_rate[0]
        logging_dict["step"] = tf.train.get_global_step()
        logging_dict.update(loss_dict)
        # if metric_update_op_dict is not None:
        #     logging_dict.update(metric_update_op_dict)
        tensor_order = logging_dict.keys()

        def format_fn(tensor_dict):
            stats = []
            for k in tensor_order:
                tensor_value = tensor_dict[k]
                stats.append("%s = %s" % (k, tensor_value))
            return ",".join(stats)

        log_step_count_steps = self.train_config.log_step_count_steps

        logging_hook = tf.train.LoggingTensorHook(
            logging_dict, every_n_iter=log_step_count_steps, formatter=format_fn)
        hooks.append(logging_hook)

        # if self.train_config.train_distribute in [
        #     DistributionStrategy.CollectiveAllReduceStrategy,
        #     DistributionStrategy.MultiWorkerMirroredStrategy
        # ]:
        #     # for multi worker strategy, we could not replace the
        #     # inner CheckpointSaverHook, so just use it.
        #     scaffold = tf.train.Scaffold()
        #     chief_hooks = []
        # else:
        #     scaffold = tf.train.Scaffold(
        #         saver=tf.train.Saver(
        #             sharded=True, max_to_keep=self.train_config.keep_checkpoint_max))
        #     # saver hook
        #     saver_hook = estimator_utils.CheckpointSaverHook(
        #         checkpoint_dir=self.model_dir,
        #         save_secs=self._config.save_checkpoints_secs,
        #         save_steps=self._config.save_checkpoints_steps,
        #         scaffold=scaffold,
        #         write_graph=True)
        #     chief_hooks = []
        #     if estimator_utils.is_chief():
        #         hooks.append(saver_hook)

        # profiling hook
        # if self.train_config.is_profiling and estimator_utils.is_chief():
        #     profile_hook = tf.train.ProfilerHook(
        #         save_steps=log_step_count_steps, output_dir=self.model_dir)
        #     hooks.append(profile_hook)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),
        )

        predict_dict = model.build_predict_graph()
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            predictions=predict_dict,
            train_op=train_op,
            # scaffold=scaffold,
            # training_chief_hooks=chief_hooks,
            training_hooks=hooks,
        )

    def _eval_model_fn(self, features, labels):
        start = time.time()
        model = self._rank_model(
            self._pipeline_config.model_config,
            self._pipeline_config.feature_configs,
            features,
            labels,
            is_training=True
        )

        predict_dict = model.build_predict_graph()
        metric_dict = model.build_metric_graph(self._pipeline_config.eval_config)

        loss_dict = model.build_loss_graph()
        loss = tf.add_n(list(loss_dict.values()))
        loss_dict["total_loss"] = loss

        for loss_key in loss_dict.keys():
            loss_tensor = loss_dict[loss_key]
            # add key-prefix to make loss metric key in the same family of train loss
            metric_dict["loss/loss/" + loss_key] = tf.metrics.mean(loss_tensor)
        logging.info("metric_dict keys: %s" % metric_dict.keys())

        end = time.time()
        logging.info("eval graph construct finished. Time %.3fs" % (end - start))
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=loss,
            predictions=predict_dict,
            eval_metric_ops=metric_dict
        )

    def _export_model_fn(self, features):
        model = self._rank_model(
            self._pipeline_config.model_config,
            self._pipeline_config.feature_configs,
            features,
            None,
            is_training=False
        )
        predict_dict = model.build_predict_graph()

        # add output info to estimator spec
        outputs = {}
        output_list = ["probs"]
        for out in output_list:
            assert out in predict_dict, \
                "output node %s not in prediction_dict, can not be exported" % out
            outputs[out] = predict_dict[out]
            logging.info(
                "output %s shape: %s type: %s" %
                (out, outputs[out].get_shape().as_list(), outputs[out].dtype))
        export_outputs = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(outputs)
        }

        # save train pipeline.config for debug purpose
        pipeline_path = os.path.join(self._model_dir, "pipeline.config")
        if tf.gfile.Exists(pipeline_path):
            tf.add_to_collection(
                tf.GraphKeys.ASSET_FILEPATHS,
                tf.constant(pipeline_path, dtype=tf.string, name="pipeline.config")
            )
        else:
            print("train pipeline_path(%s) does not exist" % pipeline_path)

        # # add more asset files
        # if "asset_files" in params:
        #     for asset_file in params["asset_files"].split(","):
        #         _, asset_name = os.path.split(asset_file)
        #         tf.add_to_collection(
        #             tf.GraphKeys.ASSET_FILEPATHS,
        #             tf.constant(asset_file, dtype=tf.string, name=asset_name))

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            loss=None,
            predictions=outputs,
            export_outputs=export_outputs
        )

    def _model_fn(self, features, labels, mode, config, params):
        os.environ["tf.estimator.mode"] = mode
        os.environ["tf.estimator.ModeKeys.TRAIN"] = tf.estimator.ModeKeys.TRAIN
        if mode == tf.estimator.ModeKeys.TRAIN:
            return self._train_model_fn(features, labels)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return self._eval_model_fn(features, labels)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return self._export_model_fn(features)
        else:
            raise ValueError("Mode:%s not supported." % mode)
