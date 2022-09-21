# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/1/25 6:38 PM
# desc:

import math
import random
from typing import List


class UniformSampling(object):
    def __init__(self, candidate_ids: List[str]):
        assert candidate_ids
        self.candidate_ids = candidate_ids

    def get(self):
        return self.candidate_ids[random.randint(0, len(self.candidate_ids)-1)]


class WeightedSampling(object):
    def __init__(self, candidate_ids: List[str], weights: List[float]):
        assert candidate_ids and weights
        assert len(candidate_ids) == len(weights)
        assert all([x > 0 for x in weights])

        self.candidate_ids = candidate_ids

        # 归一化采样概率
        sum_weights = sum(weights)
        self.probs = [x / sum_weights for x in weights]
        self.create_alias_table()

    def create_alias_table(self):
        N = len(self.probs)
        Prob = [0.0] * N
        Alias = [0] * N
        smaller, larger = [], []
        for kk, prob in enumerate(self.probs):
            Prob[kk] = N * prob
            if Prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # 通过拼凑，将各个类别都凑为1
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            Alias[small] = large
            Prob[large] = Prob[large] - (1.0 - Prob[small])  # 将大的分到小的上

            if Prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        self.alias_table_Prob = Prob
        self.alias_table_Alias = Alias

    def get(self):
        N = len(self.probs)
        kk = random.randint(0, N-1)
        if random.random() < self.alias_table_Prob[kk]:
            return self.candidate_ids[kk]
        else:
            return self.candidate_ids[self.alias_table_Alias[kk]]


class W2VSampling(WeightedSampling):
    def __init__(self, candidate_ids: List[str], weights: List[float]):
        weights = [math.pow(x, 3.0 / 4.0) for x in weights]
        super(W2VSampling, self).__init__(candidate_ids, weights)


class BaseSampler(object):
    def __init__(self, input_path: str, sampling_type="W2VSampling"):
        self.input_path = input_path
        self.load_item_attr()

        pairs = sorted(list(self.item_weight_dict.items()))
        candidate_ids = [x[0] for x in pairs]
        weights = [x[1] for x in pairs]
        if sampling_type == "W2VSampling":
            self.sampling = W2VSampling(candidate_ids, weights)
        elif sampling_type == "WeightedSampling":
            self.sampling = WeightedSampling(candidate_ids, weights)
        else:
            self.sampling = UniformSampling(candidate_ids)

    def load_user_edge_items(self, edge_input_path):
        user_edge_items_dict = dict()
        with open(edge_input_path, mode="r") as fin:
            for line in fin:
                split = line.strip().split("\t")
                user_edge_items_dict[split[0]] = list(set(split[1].split(",")))
        return user_edge_items_dict

    def load_item_attr(self):
        item_weight_dict = dict()
        item_attr_dict = dict()
        with open(self.input_path, mode="r") as fin:
            for line in fin:
                split = line.strip().split("\t")
                item_weight_dict[split[0]] = float(split[1])
                item_attr_dict[split[0]] = split[2]
        self.item_weight_dict = item_weight_dict
        self.item_attr_dict = item_attr_dict

    def get_item_fea(self, batch_item_ids: List[str]):
        return [self.item_attr_dict[x] for x in batch_item_ids]


class NegativeSampler(BaseSampler):
    """
    加权随机负采样，会排除Mini-Batch内的ItemId
    """
    def __init__(self, input_path: str, sampling_type="W2VSampling"):
        super(NegativeSampler, self).__init__(input_path, sampling_type)

    def get_samples(self, user_id: str, num_sample: int, batch_item_ids: List[str]):
        res = []
        max_try_num = num_sample + 50
        try_num = 0
        while try_num < max_try_num and len(res) < num_sample:
            tmp = self.sampling.get()
            if tmp not in res and tmp not in batch_item_ids:
                res.append(tmp)
            try_num += 1
        return self.get_item_fea(res)


class NegativeSamplerV2(BaseSampler):
    """
    加权随机负采样，会跟排除Mini-Batch内的User有边的Item Id
    """

    def __init__(self, input_path: str, pos_edge_input_path, sampling_type="W2VSampling"):
        super(NegativeSamplerV2, self).__init__(input_path, sampling_type)
        self.user_pos_items_dict = self.load_user_edge_items(pos_edge_input_path)

    def get_samples(self, user_id: str, num_sample: int, batch_item_ids: List[str]):
        res = []
        user_pos_items = self.user_pos_items_dict.get(user_id, [])
        max_try_num = num_sample + 50
        try_num = 0
        while try_num < max_try_num and len(res) < num_sample:
            tmp = self.sampling.get()
            if tmp not in res and tmp not in batch_item_ids and tmp not in user_pos_items:
                res.append(tmp)
            try_num += 1
        return self.get_item_fea(res)


class HardNegativeSampler(BaseSampler):
    """
    加权随机负采样，会排除Mini-Batch内的Item Id，同时HardNegative边表中(一般为曝光未点击)进行负采样作为HardNegative
    """
    def __init__(self, input_path: str, hard_neg_edge_input_path: str, num_hard_sample: int, sampling_type="W2VSampling"):
        super(HardNegativeSampler, self).__init__(input_path, sampling_type)
        self.user_hard_neg_items_dict = self.load_user_edge_items(hard_neg_edge_input_path)
        self.num_hard_sample = num_hard_sample

    def get_samples(self, user_id: str, num_sample: int, batch_item_ids: List[str]):
        res = []

        max_try_num = num_sample + 50
        try_num = 0
        while try_num < max_try_num and len(res) < num_sample:
            tmp = self.sampling.get()
            if tmp not in res and tmp not in batch_item_ids:
                res.append(tmp)
            try_num += 1

        user_hard_neg_items = self.user_hard_neg_items_dict.get(user_id, [])
        if user_hard_neg_items:
            hard_neg_sampling = UniformSampling(user_hard_neg_items)
            max_try_num = self.num_hard_sample + 20
            try_num = 0
            while try_num < max_try_num and len(res) < (num_sample + self.num_hard_sample):
                tmp = hard_neg_sampling.get()
                if tmp not in res and tmp not in batch_item_ids:
                    res.append(tmp)
                try_num += 1
        return self.get_item_fea(res)


class HardNegativeSamplerV2(BaseSampler):
    """
    加权随机负采样，会跟排除Mini-Batch内的User有边的Item Id，同时HardNegative边表中(一般为曝光未点击)进行负采样作为HardNegative
    """
    def __init__(self, input_path: str, pos_edge_input_path: str,
                 hard_neg_edge_input_path: str, num_hard_sample: int,
                 sampling_type="W2VSampling"):
        super(HardNegativeSamplerV2, self).__init__(input_path, sampling_type)
        self.user_pos_items_dict = self.load_user_edge_items(pos_edge_input_path)
        self.user_hard_neg_items_dict = self.load_user_edge_items(hard_neg_edge_input_path)
        self.num_hard_sample = num_hard_sample

    def get_samples(self, user_id: str, num_sample: int, batch_item_ids: List[str]):
        res = []
        user_pos_items = self.user_pos_items_dict.get(user_id, [])

        max_try_num = num_sample + 50
        try_num = 0
        while try_num < max_try_num and len(res) < num_sample:
            tmp = self.sampling.get()
            if tmp not in res and tmp not in batch_item_ids and tmp not in user_pos_items:
                res.append(tmp)
            try_num += 1

        user_hard_neg_items = self.user_hard_neg_items_dict.get(user_id, [])
        if user_hard_neg_items:
            hard_neg_sampling = UniformSampling(user_hard_neg_items)
            max_try_num = self.num_hard_sample + 20
            try_num = 0
            while try_num < max_try_num and len(res) < (num_sample + self.num_hard_sample):
                tmp = hard_neg_sampling.get()
                if tmp not in res and tmp not in batch_item_ids and tmp not in user_pos_items:
                    res.append(tmp)
                try_num += 1
        return self.get_item_fea(res)
