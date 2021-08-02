# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:37 下午
# desc:

from typing import List


class BaseConfig(object):
    def __str__(self):
        return str(self.__dict__)


class InputField(BaseConfig):
    def __init__(self, input_name: str, input_type: str):
        self.input_name = input_name
        self.input_type = input_type
        assert self.input_name, "input_name must not be empty"
        assert self.input_type in ["int", "float", "string"], "input_type must in [int,float,string]"

    @staticmethod
    def handle(data):
        res = InputField(data["input_name"], data["input_type"])
        return res


class InputConfig(BaseConfig):
    def __init__(self,
                 input_fields: List[InputField], label_fields: List[str],
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
                 embedding_name: str = None, embedding_dim: int = 32, combiner: str = "sum",
                 num_buckets: int = 0, hash_bucket_size: int = 0
                 ):
        self.input_name = input_name
        self.feature_type = feature_type

        self.raw_input_dim = raw_input_dim

        self.embedding_name = embedding_name
        self.embedding_dim = embedding_dim
        self.combiner = combiner
        self.num_buckets = num_buckets
        self.hash_bucket_size = hash_bucket_size

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
        self.seq_att_map = seq_att_map


class ConstantLearningRate(BaseConfig):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate


class ExponentialDecayLearningRate(BaseConfig):
    def __init__(self, initial_learning_rate=0.01, decay_steps=20000, decay_factor=0.95, min_learning_rate=0.0):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate


class LearningRate(BaseConfig):
    def __init__(self, learning_rate_type="exponential_decay_learning_rate",
                 constant_learning_rate=ConstantLearningRate(),
                 exponential_decay_learning_rate=ExponentialDecayLearningRate()
                 ):
        self.learning_rate_type = learning_rate_type
        self.constant_learning_rate = constant_learning_rate
        self.exponential_decay_learning_rate = exponential_decay_learning_rate


class SgdOptimizer(BaseConfig):
    def __init__(self, learning_rate=LearningRate()):
        self.learning_rate = learning_rate


class AdagradOptimizer(BaseConfig):
    def __init__(self, learning_rate=LearningRate()):
        self.learning_rate = learning_rate


class AdamOptimizer(BaseConfig):
    def __init__(self, learning_rate=LearningRate(), beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2


class Optimizer(BaseConfig):
    def __init__(self, optimizer_type="sgd_optimizer",
                 sgd_optimizer=SgdOptimizer(),
                 adagrad_optimizer=AdagradOptimizer(),
                 adam_optimizer=AdamOptimizer()
                 ):
        self.optimizer_type = optimizer_type
        self.sgd_optimizer = sgd_optimizer
        self.adagrad_optimizer = adagrad_optimizer
        self.adam_optimizer = adam_optimizer


class TrainConfig(BaseConfig):
    def __init__(self, optimizer_config,
                 log_step_count_steps=1000,
                 save_checkpoints_steps=20000,
                 keep_checkpoint_max=3
                 ):
        self.optimizer_config = optimizer_config
        self.log_step_count_steps = log_step_count_steps
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max


class EvalMetric(BaseConfig):
    def __init__(self, name):
        self.name = name


class AUC(EvalMetric):
    def __init__(self):
        super(AUC, self).__init__("auc")


class GroupAUC(EvalMetric):
    def __init__(self, gid_field: str, reduction="mean_by_sample_num"):
        """
        Args:
            gid_field: group ids, A int or string `Tensor` whose shape matches `predictions`.
            reduction: reduction method for auc of different users
                * "mean": simple mean of different users
                * "mean_by_sample_num": weighted mean with sample num of different users
                * "mean_by_positive_num": weighted mean with positive sample num of different users
        """
        super(GroupAUC, self).__init__("gauc")
        self.gid_field = gid_field
        self.reduction = reduction


class PCOPC(EvalMetric):
    def __init__(self):
        super(PCOPC, self).__init__("pcopc")


class EvalConfig(BaseConfig):
    def __init__(self,
                 auc: AUC = None,
                 gauc: GroupAUC = None,
                 pcopc: PCOPC = None
                 ):
        metric_set = []
        if auc:
            metric_set.append(auc)
        if gauc and gauc.gid_field:
            metric_set.append(gauc)
        if pcopc:
            metric_set.append(pcopc)
        self.metric_set = metric_set


class ExportConfig(BaseConfig):
    def __init__(self, export_dir):
        self.export_dir = export_dir


class DNNConfig(BaseConfig):
    def __init__(self, hidden_units: List[int],
                 activation: str = "tf.nn.relu",
                 use_bn: bool = False,
                 dropout_ratio=None
                 ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio

    @staticmethod
    def handle(data):
        res = DNNConfig(data["hidden_units"])
        if "activation" in data:
            res.activation = data["activation"]
        if "use_bn" in data:
            res.use_bn = data["use_bn"]
        if "dropout_ratio" in data:
            res.dropout_ratio = data["dropout_ratio"]
        return res


class DNNTower(BaseConfig):
    def __init__(self, input_group: str, dnn_config: DNNConfig):
        self.input_group = input_group
        self.dnn_config = dnn_config


class DINConfig(DNNConfig):
    def __init__(self, hidden_units: List[int],
                 activation: str = "tf.nn.relu",
                 use_bn: bool = False,
                 dropout_ratio=None
                 ):
        assert hidden_units and hidden_units[-1] == 1
        super(DINConfig, self).__init__(hidden_units, activation, use_bn, dropout_ratio)


class DINTower(BaseConfig):
    def __init__(self, input_group, din_config: DINConfig):
        self.input_group = input_group
        self.din_config = din_config


class BiasTower(BaseConfig):
    def __init__(self, input_group):
        self.input_group = input_group


class ModelConfig(BaseConfig):
    def __init__(self, model_class: str,
                 feature_groups: List[FeatureGroup],
                 dnn_towers: List[DNNTower] = None,
                 din_towers: List[DINTower] = None,
                 final_dnn: DNNTower = None,
                 bias_tower: BiasTower = None,
                 embedding_regularization: float = 0.0,
                 l2_regularization: float = 0.0001,
                 ):
        self.model_class = model_class
        self.feature_groups = feature_groups
        self.dnn_towers = dnn_towers
        self.din_towers = din_towers
        self.final_dnn = final_dnn
        self.bias_tower = bias_tower
        self.embedding_regularization = embedding_regularization
        self.l2_regularization = l2_regularization


class PipelineConfig(BaseConfig):
    def __init__(self, model_dir: str, input_config: InputConfig, feature_config: FeatureConfig
                 , model_config: ModelConfig
                 , train_config: TrainConfig = None, eval_config: EvalConfig = None, export_config: ExportConfig = None
                 ):
        self.model_dir = model_dir
        self.input_config = input_config
        self.feature_config = feature_config
        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.export_config = export_config

    @staticmethod
    def handle(data):
        res = PipelineConfig(data["model_dir"], data["input_config"], data["feature_config"], data["model_config"])
        if "train_config" in data:
            res.train_config = data["train_config"]
        if "eval_config" in data:
            res.eval_config = data["eval_config"]
        if "export_config" in data:
            res.export_config = data["export_config"]
        return res
