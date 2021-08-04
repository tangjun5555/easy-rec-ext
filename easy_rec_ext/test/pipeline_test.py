# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:46 下午
# desc:

import json

line_sep = "##" * 10 + "\n"


def test_01():
    print(line_sep)
    from easy_rec_ext.core.pipeline import PipelineConfig
    with open("cvr_v1_din_v1.json", "r") as f:
        tmp = json.load(f)
        res = PipelineConfig.handle(tmp)
        # print(type(tmp))
        # print(type(tmp["input_fields"]))
        # print(type(tmp["input_fields"][0]))
        print(res)
