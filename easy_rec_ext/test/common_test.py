# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/1 12:23 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

line_sep = "\n" + "##" * 20 + "\n"


def test_09():
    import oss2
    auth = oss2.Auth("LTAI4G3GfJSTZwsi8ySkYv4Z", "0EW3RLRUV7zWwxtxHiuQztl2dbAVNr")
    bucket = oss2.Bucket(auth, "http://oss-cn-hangzhou.aliyuncs.com", "shihuo-bigdata-oss")
    object_stream = bucket.get_object("tangjun0612/data/ShouyeFeedRankCvrDeepV1Data/20210711/part-00000")
    # object_stream.read()
    # print(object_stream.read(1000))
    index = 1
    while index <= 10:
        tmp = object_stream.read(1024)

        print("第%d行:%s" % (index, ))

def test_08():
    tf.disable_eager_execution()
    from tensorflow.python.client import timeline

    x = tf.random_normal([1000, 1000])
    y = tf.random_normal([1000, 1000])
    res = tf.matmul(x, y)

    # Run the graph with full trace option
    with tf.Session() as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(res, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open("timeline.json", "w") as f:
            f.write(ctf)


def test_07():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[1.0, 2.0, 3.0],
        dtype=tf.dtypes.float32
    )
    t2 = tf.expand_dims(t1, axis=1)
    t3 = tf.concat([t2, t2], axis=1)
    t4 = tf.concat([t3, t2, t1], axis=1)
    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
    print(line_sep)
    print(t3)
    print(line_sep)
    print(t4)


def test_06():
    tf.enable_eager_execution()
    from tensorflow.python.ops import sparse_ops
    t1 = tf.constant(
        value=[
            [-1],
            [0],
            [1]
        ]
    )
    t2 = sparse_ops.from_dense(t1)
    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)


def test_05():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[
            [-1],
            [0],
            [1]
        ]
    )
    t2 = tf.one_hot(t1, 5)
    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)


def test_04():
    tf.enable_eager_execution()
    # tf.disable_eager_execution()

    input_tensor = tf.constant(
        value=[
            ["431645d82843f859"],
            ["431645d82843f859"],
            [""],
            [""]
        ],
        dtype=tf.dtypes.string
    )
    num_buckets = 100

    from easy_rec_ext.utils.string_ops import string_to_hash_bucket
    res = string_to_hash_bucket(input_tensor, num_buckets)
    print(line_sep)
    print(res)
    # with tf.Session() as sess:
    #     print(line_sep)
    #     print(sess.run(input_tensor))
    #     print(line_sep)
    #     print(sess.run(res))


def test_03():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[
            ["431645d82843f859"],
            ["431645d82843f859"],
            [""],
            [""]
        ],
        dtype=tf.dtypes.string
    )
    t2 = tf.string_to_hash_bucket(t1, 1000)
    t3 = tf.string_to_hash_bucket_fast(t1, 1000)
    t4 = tf.string_to_hash_bucket_strong(t1, 1000, [555, 1234])

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
    print(line_sep)
    print(t3)
    print(line_sep)
    print(t4)

    print(line_sep)
    print(t1 == "")
    # print(line_sep)
    print(tf.ones_like(t1))


def test_02():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[[0], [1], [-1]],
        dtype=tf.dtypes.int64,
    )
    t2 = tf.sparse.from_dense(t1)

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)


def test_01():
    tf.enable_eager_execution()
    t1 = tf.constant(value=[0.0, 1.0, 0.0], dtype=tf.dtypes.float32)
    t2 = tf.cast(t1, tf.dtypes.bool)

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
