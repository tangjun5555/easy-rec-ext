# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/30 12:21 下午
# desc:

import os
import logging
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import init_ops

import tensorflow_recommenders_addons as tfra
from easy_rec_ext.utils import variable_util, string_ops

if tf.__version__ >= "2.0":
    gfile = tf.compat.v1.gfile
    tf = tf.compat.v1
else:
    gfile = tf.gfile
filename = str(os.path.basename(__file__)).split(".")[0]


def get_embedding_variable(name, dim, vocab_size=None, key_is_string=False):
    assert name
    assert isinstance(dim, int) and dim > 0

    initializer = init_ops.random_normal_initializer(mean=0.0, stddev=0.1)
    use_pretrain_variable = False
    if "pretrain_variable_dir" in os.environ and gfile.Exists(os.environ["pretrain_variable_dir"] + "/" + name):
        if "used_pretrain_variable_list" not in os.environ or not os.environ["used_pretrain_variable_list"]:
            used_pretrain_variable_list = []
        else:
            used_pretrain_variable_list = os.environ["used_pretrain_variable_list"].split(",")

        if name not in used_pretrain_variable_list:
            used_pretrain_variable_list.append(name)
            os.environ["used_pretrain_variable_list"] = ",".join(used_pretrain_variable_list)
            values = variable_util.load_variable_by_file(os.environ["pretrain_variable_dir"] + "/" + name)
            assert len(values.shape) == 2
            assert values.shape[1] == dim
            if vocab_size:
                assert values.shape[0] == vocab_size
            initializer = tf.constant(value=values, dtype=tf.dtypes.float32)
            use_pretrain_variable = True
            logging.info("%s get_embedding_variable, load %s weight from file" % (str(os.path.basename(__file__)).split(".")[0], name))

    with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
        if os.environ["use_dynamic_embedding"] == "1":
            if key_is_string:
                return tfra.dynamic_embedding.get_variable(
                    name=name,
                    key_dtype=dtypes.string,
                    value_dtype=dtypes.float32,
                    dim=dim,
                    initializer=initializer,
                )
            else:
                return tfra.dynamic_embedding.get_variable(
                    name=name,
                    key_dtype=dtypes.int64,
                    value_dtype=dtypes.float32,
                    dim=dim,
                    initializer=initializer,
                )
        else:
            if use_pretrain_variable:
                return tf.get_variable(
                    name=name,
                    dtype=dtypes.float32,
                    initializer=initializer,
                    trainable=True,
                )
            else:
                return tf.get_variable(
                    name=name,
                    shape=(vocab_size, dim),
                    dtype=dtypes.float32,
                    initializer=initializer,
                    trainable=True,
                )


def _to_sparse_ids(input_tensor):
    indices = array_ops.where_v2(
        math_ops.greater_equal(input_tensor, array_ops.zeros_like(input_tensor))
    )
    values = array_ops.gather_nd(input_tensor, indices)
    shape = array_ops.shape(input_tensor, out_type=dtypes.int64)
    return sparse_tensor.SparseTensor(indices, values, shape)


def _prune_invalid_ids(sparse_ids, sparse_weights=None):
    """
    Prune invalid IDs (< 0) from the input ids and weights.

    Args:
        sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
            ids. `d_0` is typically batch size.
        sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
            float weights corresponding to `sparse_ids`, or `None` if all weights
            are be assumed to be 1.0.

    Returns:
        Same as Args
    """
    is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
    if sparse_weights is not None:
        is_id_valid = math_ops.logical_and(
            is_id_valid,
            array_ops.ones_like(sparse_weights.values, dtype=dtypes.bool)
        )
    sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
    if sparse_weights is not None:
        sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
    return sparse_ids, sparse_weights


def _prune_invalid_weights(sparse_ids, sparse_weights=None):
    """
    Prune invalid weights (< 0) from the input ids and weights.

    Args:
        sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
            ids. `d_0` is typically batch size.
        sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
            float weights corresponding to `sparse_ids`, or `None` if all weights
            are be assumed to be 1.0.

    Returns:
        Same as Args
    """
    if sparse_weights is not None:
        is_weights_valid = math_ops.greater(sparse_weights.values, 0)
        sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_weights_valid)
        sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_weights_valid)
    return sparse_ids, sparse_weights


def safe_embedding_lookup(params, ids):
    assert ids.get_shape().as_list()[-1] == 1
    if os.environ["use_dynamic_embedding"] == "1":
        if ids.dtype == tf.dtypes.string:
            condition = string_ops.compute_valid_string_id_condition(ids)
        else:
            condition = tf.math.greater_equal(ids, 0)
            if ids.dtype != dtypes.int64:
                ids = math_ops.to_int64(ids)
        values = tfra.dynamic_embedding.embedding_lookup_unique(
            params,
            ids,
            name="tfra_embedding_lookup_unique",
        )
        values = tf.squeeze(values, axis=-2)
        condition = tf.concat([condition] * values.get_shape().as_list()[-1], axis=-1)
        zeros = tf.zeros_like(values)
        return tf.where(condition, values, zeros)
    else:
        return safe_embedding_lookup_sparse(
            params,
            _to_sparse_ids(ids),
        )


def safe_embedding_lookup_sparse(embedding_weights,
                                 sparse_ids,
                                 sparse_weights=None,
                                 combiner="mean",
                                 default_id=None,
                                 name=None,
                                 partition_strategy="div",
                                 max_norm=None):
    """Lookup embedding results, accounting for invalid IDs and empty features.

    The partitioned embedding in `embedding_weights` must all be the same shape
    except for the first dimension. The first dimension is allowed to vary as the
    vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
    may be a `PartitionedVariable` as returned by using `tf.get_variable()` with a
    partitioner.

    Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
    with non-positive weight. For an entry with no features, the embedding vector
    for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

    The ids and weights may be multi-dimensional. Embeddings are always aggregated
    along the last dimension.

    Args:
      embedding_weights:  A list of `P` float `Tensor`s or values representing
          partitioned embedding `Tensor`s.  Alternatively, a `PartitionedVariable`
          created by partitioning along dimension 0.  The total unpartitioned
          shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
          vocab size and `e_1, ..., e_m` are the embedding dimensions.
      sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
          ids. `d_0` is typically batch size.
      sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
          float weights corresponding to `sparse_ids`, or `None` if all weights
          are be assumed to be 1.0.
      combiner: A string specifying how to combine embedding results for each
          entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
          the default.
      default_id: The id to use for an entry with no features.
      name: A name for this operation (optional).
      partition_strategy: A string specifying the partitioning strategy.
          Currently `"div"` and `"mod"` are supported. Default is `"div"`.
      max_norm: If not `None`, all embeddings are l2-normalized to max_norm before
          combining.


    Returns:
      Dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

    Raises:
      ValueError: if `embedding_weights` is empty.
    """
    if embedding_weights is None:
        raise ValueError("Missing embedding_weights %s." % embedding_weights)

    # embed_tensors = [ops.convert_to_tensor(embedding_weights)]
    embed_tensors = [embedding_weights]
    with ops.name_scope(name, default_name="embedding_lookup",
                        values=embed_tensors + [sparse_ids, sparse_weights]) as scope:
        # Reshape higher-rank sparse ids and weights to linear segment ids.
        original_shape = sparse_ids.dense_shape
        original_rank_dim = sparse_ids.dense_shape.get_shape()[0]
        # original_rank = (
        #     array_ops.size(original_shape)
        #     if original_rank_dim.value is None else original_rank_dim.value)
        original_rank = array_ops.size(original_shape)
        sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
            math_ops.reduce_prod(
                array_ops.slice(original_shape, [0], [original_rank - 1])),
            array_ops.gather(original_shape, original_rank - 1)
        ])
        if sparse_weights is not None:
            sparse_weights = sparse_tensor.SparseTensor(sparse_ids.indices,
                                                        sparse_weights.values,
                                                        sparse_ids.dense_shape)

        # Prune invalid ids and weights.
        sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)
        if combiner != "sum":
            sparse_ids, sparse_weights = _prune_invalid_weights(
                sparse_ids, sparse_weights)

        # Fill in dummy values for empty features, if necessary.
        sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(
            sparse_ids, default_id or 0)
        if sparse_weights is not None:
            sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

        indices = sparse_ids.indices
        values = sparse_ids.values
        if values.dtype != dtypes.int64:
            values = math_ops.to_int64(values)
        sparse_ids = sparse_tensor.SparseTensor(
            indices=indices, values=values, dense_shape=sparse_ids.dense_shape)

        result = embedding_ops.embedding_lookup_sparse(
            embedding_weights,
            sparse_ids,
            sparse_weights,
            combiner=combiner,
            partition_strategy=partition_strategy,
            # name=None if default_id is None else scope,
            max_norm=max_norm
        )

        if default_id is None:
            # Broadcast is_row_empty to the same shape as embedding_lookup_result,
            # for use in Select.
            is_row_empty = array_ops.tile(
                array_ops.reshape(is_row_empty, [-1, 1]),
                array_ops.stack([1, array_ops.shape(result)[1]]))

            result = array_ops.where(
                is_row_empty, array_ops.zeros_like(result), result, name=scope)

        # Reshape back from linear ids back into higher-dimensional dense result.
        final_result = array_ops.reshape(
            result,
            array_ops.concat([
                array_ops.slice(
                    math_ops.cast(original_shape, dtypes.int32), [0],
                    [original_rank - 1]),
                array_ops.slice(array_ops.shape(result), [1], [-1])
            ], 0))
        final_result.set_shape(
            tensor_shape.unknown_shape(
                (original_rank_dim - 1)).concatenate(result.get_shape()[1:]))
        return final_result
