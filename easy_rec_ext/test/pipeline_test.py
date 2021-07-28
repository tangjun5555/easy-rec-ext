# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:46 下午
# desc:

import json

line_sep = "##" * 10 + "\n"


def test_01():
  print(line_sep)
  from easy_rec_ext.core.pipeline import InputConfig
  with open("input_config.json", "r") as f:
    tmp = json.load(f)
    res = InputConfig.handle(tmp)
    print(type(tmp))
    print(type(tmp["input_fields"]))
    print(type(tmp["input_fields"][0]))
    print(res)

  # print(line_sep)
  # with open("input_config.json", "r") as f:
  #   res = json.load(f, object_hook=InputConfig.handle)
  #   print(res)
