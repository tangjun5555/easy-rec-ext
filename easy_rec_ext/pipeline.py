# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:37 下午
# desc:

from typing import List


class BaseConfig(object):
  def __str__(self):
    return str(self.__dict__)


class InputField(BaseConfig):
  def __init__(self, input_name: str, input_type: str = "string"):
    self.input_name = input_name
    self.input_type = input_type

  @staticmethod
  def handle(data):
    res = InputField(data["input_name"])
    if "input_type" in data:
      res.input_type = data["input_type"]
    return res


class InputConfig(BaseConfig):
  def __init__(self,
               input_fields: List[InputField], label_fields: List[str] = None,
               input_type: str = "CSV",
               train_input_path: List[str] = None, eval_input_path: List[str] = None,
               num_epochs: int = 2, batch_size: int = 256,
               ):
    self.input_fields = input_fields
    self.label_fields = label_fields
    self.input_type = input_type
    self.num_epochs = num_epochs
    self.batch_size = batch_size

  @staticmethod
  def handle(data):
    print("InputConfig handle, data:", str(data))
    input_fields = []
    for input_field in data["input_fields"]:
      input_fields.append(InputField.handle(input_field))
    res = InputConfig(input_fields, data["label_fields"])
    if "input_type" in data:
      res.input_type = data["input_type"]
    if "num_epochs" in data:
      res.num_epochs = data["num_epochs"]
    if "batch_size" in data:
      res.batch_size = data["batch_size"]
    return res


class TrainConfig(BaseConfig):
  def __init__(self):
    pass


class EvalConfig(BaseConfig):
  def __init__(self):
    pass


class ExportConfig(BaseConfig):
  def __init__(self):
    pass


class FeatureConfig(BaseConfig):
  def __init__(self):
    pass


class ModelConfig(BaseConfig):
  def __init__(self):
    pass


class PipelineConfig(BaseConfig):
  def __init__(self):
    pass
