# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/6/27 0:19
# desc:

import random
import time
import argparse
from annoy import AnnoyIndex
import numpy as np
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

parser = argparse.ArgumentParser()
parser.add_argument("--model_pb_path", type=str, required=True)
parser.add_argument("--item_fea_path", type=str, required=True)
parser.add_argument("--eval_sample_path", type=str, required=True)
parser.add_argument("--vector_dim", type=int, required=False, default=100)
parser.add_argument("--topks", type=str, required=False, default="5,20,100")
parser.add_argument("--default_user_fea", type=str, required=False, default="1000,2340930|1409153|2340930|3214150|308742|3877041|3532002|2412328|3872382|3532002|4599135|694225|4444531|2339773|620939|2663756|4064135|2663756|4778761|2033067|1922292|1907674|4240134|2196555|1109842|1373387|1520357|4720367|1822774|462579|543879|1822774|2033067|4950759|203347|848341|843246|4827739|4793529|1925478|2153891|3255674|2594546|4185482|4487355|3011372|4402665|2956844|544269|2956844,1051370|5071267|1051370|5071267|1864538|5053508|1299190|1879194|1879194|1299190|245312|807138|1851156|1851156|2352202|1299190|1299190|1299190|2982027|3579754|4993094|1851156|2939262|2430608|3579754|1202097|3579754|3579754|3579754|3579754|3579754|3579754|3579754|411153|1851156|5071267|1879194|1051370|238434|171529|1051370|2440115|1851156|1653613|1653613|1349561|1789614|2419959|2419959|2419959,1|1|4|1|1|1|1|1|4|1|1|1|1|1|1|1|4|1|1|1|4|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|3|4|1|1|1|1|1|1|4|1|1|1|1|4,1|1|1|1|1|2|2|2|2|2|2|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|4|5|5|5|5|5|6|6|6|6|6|6|6,1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50")
parser.add_argument("--default_item_fea", type=str, required=False, default="1585986,1299190")
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
        with open(args.item_fea_path, mode="r") as f:
            for line in f:
                split = line.strip().split("#")
                assert len(split) == 2, line
                all_item_ids.append(split[0])
                batch_fea_input.append(split[1])
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
        annoy_index.build(50)
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
    topks = [int(x) for x in args.topks.split(",")]
    recall_rates = [0.0 for i in range(len(topks))]
    with open(args.eval_sample_path, mode="r") as f:
        sample_num = 0
        for line in f:
            sample_num += 1
            split = line.strip().split(",")
            labels = [split[-2]]
            user_vector = processor.build_user_vector(",".join(split[1:-2]))
            preds = processor.search_recall_items(
                user_vector,
                max(topks),
            )
            if sample_num % 1000 == 0:
                print(sample_num, line)
                print(labels, preds)
                print(user_vector)
            for i in range(len(topks)):
                recall_rates[i] += recall_at_k(labels, preds[:topks[i]])
    print("评估样本数量:", sample_num)
    for i in range(len(topks)):
        print("评估指标recall@%d:%s" % (topks[i], str(recall_rates[i] / sample_num)))
