# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/6/27 0:19
# desc:

import time
import random
import argparse
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex
import tensorflow_recommenders_addons as tfra

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", type=str, required=True, choices=["predict", "evaluate"], help="任务类型")
parser.add_argument("--model_pb_path", type=str, required=True, help="模型pb格式地址")
parser.add_argument("--item_input_path", type=str, required=True,
                    help="Item表，Schema为: itemid:string\tweight:float\titem_attrs:string",
                    )
parser.add_argument("--pos_input_path", type=str, required=False, default=None,
                    help="正样本表，Schema为: label:int,features",
                    )
parser.add_argument("--predict_input_path", type=str, required=False, default=None,
                    help="预测表，Schema为: sample_id:string\tfeatures",
                    )
parser.add_argument("--predict_num", type=int, required=False, default=50,
                    help="预测数量",
                    )
parser.add_argument("--item_fea_num", type=int, required=True, help="物品特征数量")
parser.add_argument("--default_user_fea", type=str, required=True, help="默认用户特征")
parser.add_argument("--default_item_fea", type=str, required=True, help="默认物品特征")
parser.add_argument("--vector_dim", type=int, required=True, help="向量维度")
parser.add_argument("--topks", type=str, required=False, default="5,20,100")
args = parser.parse_args()
print("Run params:" + str(args))


class Processor(object):
    def load_model(self):
        sess = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(sess, ["serve"], args.model_pb_path)
        self.sess = sess
        print("%s加载模型" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def build_user_vector(self, user_fea):
        features = np.array(
            [user_fea + "," + args.default_item_fea]
        )
        input_tensor = self.sess.graph.get_tensor_by_name("features:0")
        output_tensor = self.sess.graph.get_tensor_by_name("user_vector:0")
        return list(self.sess.run(output_tensor, feed_dict={input_tensor: features}))[0]

    def build_item_vectors(self, item_fea_list):
        features = np.array(
            [args.default_user_fea + "," + item_fea for item_fea in item_fea_list]
        )
        input_tensor = self.sess.graph.get_tensor_by_name("features:0")
        output_tensor = self.sess.graph.get_tensor_by_name("item_vector:0")
        return list(self.sess.run(output_tensor, feed_dict={input_tensor: features}))

    def build_annoy_index(self):
        annoy_index = AnnoyIndex(args.vector_dim, "dot")
        all_item_ids = []

        batch_fea_input = []
        i = 0
        with open(args.item_input_path, mode="r") as f:
            for line in f:
                split = line.strip().split("\t")
                assert len(split) == 3, line
                all_item_ids.append(split[0])
                batch_fea_input.append(split[2])
                if len(batch_fea_input) >= 500:
                    batch_item_vectors = self.build_item_vectors(batch_fea_input)
                    for vector in batch_item_vectors:
                        annoy_index.add_item(i, vector)
                        i += 1
                    print(i, vector)
                    batch_fea_input = []
            if batch_fea_input:
                batch_item_vectors = self.build_item_vectors(batch_fea_input)
                for vector in batch_item_vectors:
                    annoy_index.add_item(i, vector)
                    i += 1
                batch_fea_input = []

        annoy_index.build(500)
        self.all_item_ids = all_item_ids
        self.annoy_index = annoy_index
        print("%s构建索引" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    def __init__(self):
        self.load_model()
        self.build_annoy_index()

    def search_recall_items(self, user_vector, recall_num):
        ids, scores = self.annoy_index.get_nns_by_vector(user_vector, recall_num, include_distances=True)
        res = [self.all_item_ids[i] for i in ids]
        if random.randint(1, 1000) == 5:
            print("search_recall_items:")
            print(res)
            print(scores)
        return res


def recall_at_k(labels, preds):
    res = 0
    for x in preds:
        if x in labels:
            res += 1
    return res / len(labels)


if __name__ == "__main__":
    processor = Processor()

    if args.task_type == "evaluate":
        topks = [int(x) for x in args.topks.split(",")]
        recall_rates = [0.0 for i in range(len(topks))]
        with open(args.pos_input_path, mode="r") as f:
            sample_num = 0
            for line in f:
                sample_num += 1
                split = line.strip().split(",")

                user_fea = ",".join(split[1:-args.item_fea_num])
                labels = [split[-args.item_fea_num]]

                user_vector = processor.build_user_vector(user_fea)
                preds = processor.search_recall_items(
                    user_vector,
                    max(topks),
                )

                for i in range(len(topks)):
                    recall_rates[i] += recall_at_k(labels, preds[:topks[i]])

                if sample_num % 1000 == 0:
                    print(sample_num, line)
                    print(user_vector)
                    print(labels, preds)

        print("评估样本数量:", sample_num)
        for i in range(len(topks)):
            print("评估指标recall@%d:%s" % (topks[i], str(recall_rates[i] / sample_num)))

    else:
        fin = open(args.predict_input_path, mode="r")
        fout = open(args.predict_input_path + "_output", mode="w")

        sample_num = 0
        for line in fin:
            sample_num += 1
            split = line.strip().split("\t")
            sample_id = split[0]
            user_fea = split[1]

            user_vector = processor.build_user_vector(user_fea)
            preds = processor.search_recall_items(
                user_vector,
                args.predict_num,
            )
            fout.write(sample_id + "\t" + ",".join(preds) + "\n")

            if sample_num % 1000 == 0:
                print(sample_num, line)
                print(user_vector)
                print(preds)

        fout.close()
        fin.close()
