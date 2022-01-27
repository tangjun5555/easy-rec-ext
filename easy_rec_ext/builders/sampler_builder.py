# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2022/1/25 6:38 PM
# desc:


class BaseSampler(object):
    pass


class GlobalRandomNegativeSampler(BaseSampler):
    """
    全局随机负采样
    """
    pass


class InBatchNegativeSampler(BaseSampler):
    """
    in-batch负采样
    """
    pass


class CrossBatchNegativeSampler(BaseSampler):
    """
    cross-batch负采样
    """
    pass
