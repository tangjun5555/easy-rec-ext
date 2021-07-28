# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:37 下午
# desc:

from typing import List
from easy_rec_ext.layers.dnn import DNNConfig
from easy_rec_ext.model.din import DINConfig


class BaseConfig(object):
    def __str__(self):
        return str(self.__dict__)


class InputField(BaseConfig):
    def __init__(self, input_name: str, input_type: str = "int"):
        self.input_name = input_name
        self.input_type = input_type
        assert self.input_name, "input_name must not be empty"
        assert self.input_type in ["int", "float", "string"], "input_type must in [int,float,string]"

    @staticmethod
    def handle(data):
        res = InputField(data["input_name"])
        if "input_type" in data:
            res.input_type = data["input_type"]
        return res


class InputConfig(BaseConfig):
    def __init__(self,
                 input_fields: List[InputField], label_fields: List[str] = None,
                 input_type: str = "csv",
                 train_input_path: str = None, eval_input_path: str = None,
                 num_epochs: int = 2, batch_size: int = 256
                 ):
        self.input_fields = input_fields
        self.label_fields = label_fields
        self.input_type = input_type
        self.train_input_path = train_input_path
        self.eval_input_path = eval_input_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    @staticmethod
    def handle(data):
        input_fields = []
        for input_field in data["input_fields"]:
            input_fields.append(InputField.handle(input_field))
        res = InputConfig(input_fields, data["label_fields"])
        if "input_type" in data:
            res.input_type = data["input_type"]
        if "train_input_path" in data:
            res.train_input_path = data["train_input_path"]
        if "eval_input_path" in data:
            res.eval_input_path = data["eval_input_path"]
        if "num_epochs" in data:
            res.num_epochs = data["num_epochs"]
        if "batch_size" in data:
            res.batch_size = data["batch_size"]
        return res


class FeatureField(BaseConfig):
    def __init__(self, input_name: str, feature_type: str,
                 raw_input_dim=1,
                 embedding_name: str = None, embedding_dim: int = 32, num_buckets: int = 1000000, combiner: str = "sum",
                 ):
        self.input_name = input_name
        self.feature_type = feature_type
        self.raw_input_dim = raw_input_dim
        self.embedding_name = embedding_name
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.combiner = combiner

        assert self.feature_type in ["IdFeature", "RawFeature", "SequenceFeature"]


class FeatureConfig(BaseConfig):
    def __init__(self, feature_fields: List[FeatureField]):
        self.feature_fields = feature_fields


class SeqAttMap(BaseConfig):
    def __init__(self, key: str, hist_seq: str):
        self.key = key
        self.hist_seq = hist_seq


class FeatureGroup(BaseConfig):
    def __init__(self, group_name: str, feature_names: List[str] = None, seq_att_map: SeqAttMap = None):
        self.group_name = group_name
        self.feature_names = feature_names


class TrainConfig(BaseConfig):
    def __init__(self):
        pass


class EvalConfig(BaseConfig):
    def __init__(self):
        pass


class ExportConfig(BaseConfig):
    def __init__(self, export_dir):
        self.export_dir = export_dir


class DNNTower(BaseConfig):
    def __init__(self, input_group: str, dnn_config: DNNConfig):
        self.input_group = input_group
        self.dnn_config = dnn_config


class DINTower(BaseConfig):
    def __init__(self, input_group, din_config: DINConfig):
        self.input_group = input_group
        self.din_config = din_config


class BiasTower(BaseConfig):
    def __init__(self, input_group):
        self.input_group = input_group


class ModelConfig(BaseConfig):
    def __init__(self, model_class: str,
                 feature_groups: List[str],
                 dnn_towers: List[DNNTower],
                 din_towers: List[DINTower],
                 final_dnn: DNNTower,
                 bias_tower: BiasTower,
                 ):
        self.model_class = model_class
        self.feature_groups = feature_groups
        self.dnn_towers = dnn_towers
        self.din_towers = din_towers
        self.final_dnn = final_dnn
        self.bias_tower = bias_tower


class PipelineConfig(BaseConfig):
    def __init__(self, model_dir: str, input_config: InputConfig, feature_config: FeatureConfig
                 , train_config: TrainConfig = None, eval_config: EvalConfig = None, export_config: ExportConfig = None
                 , model_config: ModelConfig = None
                 ):
        self.model_dir = model_dir
        self.input_config = input_config
        self.feature_config = feature_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.export_config = export_config
        self.model_config = model_config

    @staticmethod
    def handle(data):
        res = PipelineConfig(data["model_dir"], data["input_config"], data["feature_config"])
        if "train_config" in data:
            res.train_config = data["train_config"]
        if "eval_config" in data:
            res.eval_config = data["eval_config"]
        if "export_config" in data:
            res.export_config = data["export_config"]
        return res
