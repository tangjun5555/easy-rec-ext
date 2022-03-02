# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/8/1 12:23 下午
# desc:

import os
import numpy as np
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

line_sep = "\n" + "##" * 20 + "\n"


def test_32():
    from easy_rec_ext.utils import string_ops
    input_tensor = tf.constant(
        value=[
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"
        ],
        dtype=tf.dtypes.string,
    )
    r1 = string_ops.string_to_hash_bucket(input_tensor, 50)
    print(r1)


def test_31():
    tf.disable_eager_execution()
    x = tf.placeholder(tf.float32, [3, 2])
    y1 = tf.placeholder(tf.float32, [3, 3])
    y2 = tf.placeholder(tf.float32, [3, 4])
    w1 = tf.Variable(tf.ones([2, 3]))
    w2 = tf.Variable(tf.ones([3, 4]))

    hidden = tf.matmul(x, w1)
    # hidden = tf.stop_gradient(tf.matmul(x, w1))
    # output = tf.matmul(tf.stop_gradient(hidden), w2)
    loss = tf.reduce_sum(tf.matmul(hidden, w2) - y2) + tf.reduce_sum(hidden - y1)
    # loss = tf.reduce_sum(tf.matmul(tf.stop_gradient(hidden), w2) - y2) + tf.reduce_sum(hidden - y1)
    train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("---Before Gradient Descent---")
        print("w1:\n", w1.eval(), "\nw2:\n", w2.eval())
        sess.run(train_op, feed_dict={x: np.ones(shape=(3, 2)),
                                      y1: np.ones(shape=(3, 3)),
                                      y2: np.ones(shape=(3, 4))
                                      }
                 )
        print("---After Gradient Descent---")
        print("w1:\n", w1.eval(), "\nw2:\n", w2.eval())


def test_30():
    tf.disable_eager_execution()
    label = tf.constant(
        value=[
            [1],
            [1],
            [1],
            [1]
        ],
        dtype=tf.dtypes.int64,
    )
    probs = tf.constant(
        value=[
            [0.1],
            [0.2],
            [0.3],
            [0.4]
        ]
    )

    with tf.Session() as sess:
        r1 = tf.metrics.recall_at_k(label, probs, k=2)
        r2 = tf.metrics.recall_at_k(label, probs, k=1)
        print(sess.run(r1))
        print(sess.run(r2))


def test_29():
    t1 = tf.constant(
        value=[
            [1.0] * 4,
            [2.0] * 4,
            [3.0] * 4
        ],
        dtype=tf.dtypes.float32,
    )

    r1 = t1 * 10.0
    r2 = tf.nn.dropout(t1, keep_prob=0.5)

    print(t1)
    print(r1)
    print(r2)


def test_28():
    t1 = tf.constant(
        value=[
            [-2 ** 32 + 1, 1, -2 ** 32 + 1],
            [-2 ** 32 + 1, 5, -2 ** 32 + 1],
            [-2 ** 32 + 1, 210, -2 ** 32 + 1],
        ],
        dtype=tf.dtypes.float32,
    )
    r1 = tf.nn.softmax(t1)

    print(t1)
    print(r1)


def test_27():
    t1 = tf.constant(
        value=[
            [1.0] * 4,
            [2.0] * 4,
            [3.0] * 4
        ],
        dtype=tf.dtypes.float32,
    )
    t2 = tf.constant(
        # value=[[10.0], [20.0], [30.0]],
        value=[10, 20, 30],
        dtype=tf.dtypes.float32,
    )
    r1 = tf.math.multiply(t1, t2)

    print(t1)
    print(t2)
    print(r1)


def test_26():
    t1 = tf.constant(
        value=[1.0, 2.0, 3.0],
        dtype=tf.dtypes.float32,
    )
    t2 = tf.constant(
        value=[1.0, 2.0, 3.0],
        dtype=tf.dtypes.float32,
    )
    r1 = t1 * t2
    print(r1)


def test_25():
    tf.disable_eager_execution()
    from easy_rec_ext.utils import variable_util

    t1 = variable_util.get_normal_variable(
        scope="esmm", name="t1", shape=1,
    )

    t2 = variable_util.get_normal_variable(
        scope="esmm", name="t2", shape=[1],
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(t1))
        print(sess.run(t2))


def test_24():
    t1 = tf.constant(
        value=[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=tf.dtypes.float32
    )
    t2 = tf.expand_dims(t1, axis=1)

    t3 = tf.constant(
        value=[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=tf.dtypes.float32
    )
    t4 = tf.expand_dims(t3, axis=1)

    r1 = tf.concat([t2, t4], axis=1)

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)

    print(line_sep)
    print(r1)


def test_23():
    t1 = tf.constant(
        value=[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ],
        dtype=tf.dtypes.float32
    )
    t2 = tf.constant(
        value=[
            [10.0] * 5,
            [10.0] * 5
        ],
        dtype=tf.dtypes.float32
    )
    r1 = tf.multiply(tf.expand_dims(t1, axis=-1), t2)
    r2 = tf.reshape(r1, [-1, 2 * 5])
    r3 = tf.reshape(r1, [-1, 2, 5])

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
    print(line_sep)
    print(r1)
    print(line_sep)
    print(r2)
    print(line_sep)
    print(r3)


def test_22():
    t1 = tf.constant(
        value=[
            [[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]],
            [[7, 7, 7], [8, 8, 8]]
        ],
        dtype=tf.dtypes.int32
    )
    print(line_sep)
    print(t1)

    r1 = tf.slice(t1, [0, 1, 0], [-1, 1, -1])
    print(line_sep)
    print(r1)


def test_21():
    t1 = tf.constant(
        value=[
            [1] * 4,
            [2] * 4,
            [3] * 4,
        ],
        dtype=tf.dtypes.int32,
    )
    t2 = tf.constant(
        value=[
            [4] * 4,
            [5] * 4,
            [6] * 4,
        ],
        dtype=tf.dtypes.int32,
    )
    r1 = tf.stack([t1, t2], axis=1)

    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
    print(line_sep)
    print(r1)


def test_20():
    from easy_rec_ext.utils.string_ops import string_to_hash_bucket

    t1 = tf.constant(
        value=[
            ["阿迪达斯 外套"],
            ["aj"],
            ["aj1"],
            ["aj3"],
            ["aj5"],
        ],
        dtype=tf.dtypes.string,
    )

    for i in range(100):
        r1 = string_to_hash_bucket(t1, 300 * 10000)
        print(line_sep)
        print(r1)


def test_19():
    seq_len = tf.constant(
        value=[[5], [4], [3], [2], [1], [0]],
        dtype=tf.dtypes.int32,
    )
    print(line_sep)
    print(seq_len)

    mask = tf.sequence_mask(seq_len, maxlen=5, dtype=tf.dtypes.float32)
    print(line_sep)
    print(mask)

    mask = tf.transpose(mask, perm=(0, 2, 1))
    print(line_sep)
    print(mask)

    mask = tf.tile(mask, [1, 1, 4])
    print(line_sep)
    print(mask)


def test_18():
    t1 = tf.constant(
        value=[[1], [2], [3]],
        dtype=tf.dtypes.int32,
    )
    t2 = tf.constant(
        value=[[4], [5], [6]],
        dtype=tf.dtypes.int32,
    )
    r1 = tf.concat([t1, t2], axis=-1)
    print(r1)


def test_17():
    print(str(os.path.basename(__file__)).split(".")[0])


def test_16():
    t1 = tf.constant(
        value=[
            [1],
            [2],
            [3],
        ],
        dtype=tf.dtypes.int32,
    )
    t2 = t1 + 1
    print(line_sep)
    print(t2)


def test_15():
    tf.disable_eager_execution()

    with tf.variable_scope("variable", reuse=tf.AUTO_REUSE):
        t1 = tf.get_variable("alpha", [5],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32,
                             )
        t2 = tf.get_variable("alpha", [5],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32,
                             )
    # print(line_sep + "t1")
    # print(t1)
    # print(line_sep + "t2")
    # print(t2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(t1))
        # print(sess.run(t2))
        print(t1)
        print(t2)


def test_14():
    t1 = tf.constant(
        value=[
            [2],
            [3],
        ],
        dtype=tf.dtypes.float32,
    )
    print(line_sep)
    print(t1)

    t2 = t1 + 0.1
    print(line_sep)
    print(t2)


def test_13():
    t1 = tf.constant(
        value=[
            [2],
            [3],
        ],
        dtype=tf.dtypes.int32,
    )
    print(line_sep)
    print(t1)

    t2 = tf.sequence_mask(t1, maxlen=5, dtype=tf.dtypes.float32)
    print(line_sep + "t2")
    print(t2)

    t3 = tf.transpose(t2, perm=(0, 2, 1))
    print(line_sep)
    print(t3)

    t4 = tf.tile(t3, [1, 1, 4])
    print(line_sep)
    print(t4)


def test_12():
    t1 = tf.sequence_mask(lengths=[1, 3, 2], maxlen=5)
    print(line_sep)
    print(t1)

    t2 = tf.sequence_mask(lengths=[1, 3, 2], maxlen=5, dtype=tf.dtypes.float32)
    print(line_sep)
    print(t2)


def test_11():
    t1 = tf.constant(
        value=[1, 2],
        dtype=tf.dtypes.int64,
    )
    print(t1.get_shape().as_list())


def test_10():
    print(line_sep)
    import oss2
    auth = oss2.Auth("LTAI4G3GfJSTZwsi8ySkYv4Z", "0EW3RLRUV7zWwxtxHiuQztl2dbAVNr")
    bucket = oss2.Bucket(auth, "http://oss-cn-hangzhou.aliyuncs.com", "shihuo-bigdata-oss")

    for obj in oss2.ObjectIterator(bucket, prefix='tangjun0612/data/ShouyeFeedRankCvrDeepV1V2Data/20210810/part',
                                   delimiter='/'):
        # 通过is_prefix方法判断obj是否为文件夹。
        if obj.is_prefix():  # 判断obj为文件夹。
            print('directory: ' + obj.key)
        else:  # 判断obj为文件。
            print('file: ' + obj.key)


def test_09():
    print(line_sep)

    import time
    import oss2
    auth = oss2.Auth("LTAI4G3GfJSTZwsi8ySkYv4Z", "0EW3RLRUV7zWwxtxHiuQztl2dbAVNr")
    bucket = oss2.Bucket(auth, "http://oss-cn-hangzhou.aliyuncs.com", "shihuo-bigdata-oss")

    # object_stream = bucket.get_object("tangjun0612/data/ShouyeFeedRankCvrDeepV1V2Data/20210810/part-00000")

    def generator_fn():
        index = 0
        for path in ["tangjun0612/data/ShouyeFeedRankCvrDeepV1V2Data/20210810/part-00000_50"]:
            object_stream = bucket.get_object(path)
            buffer = ""
            while True:
                tmp = str(object_stream.read(1024), encoding="utf-8")
                if not tmp:
                    break
                buffer += tmp
                if "\n" in buffer:
                    split = buffer.split("\n")
                    for i in range(len(split) - 1):
                        line = split[i]
                        index += 1
                        # print("第%d行:%s" % (index, line))
                        print("第%d行" % (index))
                        yield line
                    buffer = split[-1]

    for x in generator_fn():
        print(x)

    # buffer = ""
    # index = 0
    # while True:
    #     tmp = str(object_stream.read(1024), encoding="utf-8")
    #     if not tmp:
    #         break
    #     buffer += tmp
    #     if "\n" in buffer:
    #         split = buffer.split("\n")
    #         for i in range(len(split) - 1):
    #             line = split[i]
    #             index += 1
    #             print("第%d行:%s" % (index, line))
    #         buffer = split[-1]
    #         time.sleep(0.1)


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
    t3 = tf.squeeze(t2, axis=1)
    print(line_sep)
    print(t1)
    print(line_sep)
    print(t2)
    print(line_sep)
    print(t3)


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
    # t1 = tf.constant(
    #     value=[
    #         ["431645d82843f859"],
    #         ["431645d82843f859"],
    #         [""],
    #         [""]
    #     ],
    #     dtype=tf.dtypes.string,
    # )
    t1 = tf.constant(
        value=[
            [1],
            [-1],
            [0],
            [1]
        ],
        dtype=tf.dtypes.int64,
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


if __name__ == '__main__':
    test_09()
