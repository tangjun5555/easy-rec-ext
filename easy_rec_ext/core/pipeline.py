# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:37 下午
# desc:

from typing import List
from easy_rec_ext.layers.dnn import DNNConfig, DNNTower
from easy_rec_ext.layers.sequence_pooling import SequencePoolingConfig
from easy_rec_ext.layers.interaction import InteractionConfig
from easy_rec_ext.model.din import DINTower
from easy_rec_ext.model.bst import BSTTower
from easy_rec_ext.model.dien import DIENTower
from easy_rec_ext.model.esmm import ESMMModelConfig
from easy_rec_ext.model.esmm_v2 import ESMMV2ModelConfig
from easy_rec_ext.model.aitm import AITMModelConfig


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


class OSSConfig(BaseConfig):
    def __init__(self,
                 endpoint, bucket_name,
                 access_key_id, access_key_secret,
                 read_per_size=16 * 1024,  # length of string of per read
                 ):
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.read_per_size = read_per_size

    @staticmethod
    def handle(data):
        res = OSSConfig(data["endpoint"], data["bucket_name"], data["access_key_id"], data["access_key_secret"])
        if "read_per_size" in data:
            res.read_per_size = data["read_per_size"]
        return res


class InputConfig(BaseConfig):
    def __init__(self,
                 input_fields: List[InputField], label_fields: List[str],
                 input_type: str = "csv", oss_config: OSSConfig = None,
                 train_input_path: str = None, eval_input_path: str = None,
                 num_epochs: int = 2, batch_size: int = 256
                 ):
        self.input_fields = input_fields
        self.label_fields = label_fields

        self.input_type = input_type
        self.oss_config = oss_config

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
        if "oss_config" in data:
            res.oss_config = OSSConfig.handle(data["oss_config"])

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
    def __init__(self,
                 input_name: str, feature_type: str,
                 raw_input_dim: int = 1, raw_input_embedding_type: str = None,
                 one_hot: int = 0,
                 embedding_name: str = None, embedding_dim: int = 16,
                 num_buckets: int = 0, hash_bucket_size: int = 0,
                 sequence_pooling_config: SequencePoolingConfig = None,
                 ):
        self.input_name = input_name
        self.feature_type = feature_type

        self.raw_input_dim = raw_input_dim
        self.raw_input_embedding_type = raw_input_embedding_type

        self.one_hot = one_hot

        self.embedding_name = embedding_name if embedding_name else input_name + "_embedding"
        self.embedding_dim = embedding_dim

        self.num_buckets = num_buckets
        self.hash_bucket_size = hash_bucket_size

        self.sequence_pooling_config = sequence_pooling_config

        assert self.feature_type in ["IdFeature", "RawFeature", "SequenceFeature"]

    @staticmethod
    def handle(data):
        res = FeatureField(data["input_name"], data["feature_type"])

        if "raw_input_dim" in data:
            res.raw_input_dim = data["raw_input_dim"]
        if "raw_input_embedding_type" in data:
            res.raw_input_embedding_type = data["raw_input_embedding_type"]

        if "one_hot" in data:
            res.one_hot = data["one_hot"]

        if "embedding_name" in data:
            res.embedding_name = data["embedding_name"] if data["embedding_name"] else res.input_name + "_embedding"
        if "embedding_dim" in data:
            res.embedding_dim = data["embedding_dim"]

        if "num_buckets" in data:
            res.num_buckets = data["num_buckets"]
        if "hash_bucket_size" in data:
            res.hash_bucket_size = data["hash_bucket_size"]

        if "sequence_pooling_config" in data:
            res.sequence_pooling_config = SequencePoolingConfig.handle(data["sequence_pooling_config"])
        return res


class FeatureConfig(BaseConfig):
    def __init__(self, feature_fields: List[FeatureField]):
        self.feature_fields = feature_fields

    @staticmethod
    def handle(data):
        feature_fields = []
        for feature_field in data["feature_fields"]:
            feature_fields.append(FeatureField.handle(feature_field))
        res = FeatureConfig(feature_fields)
        return res


class SeqAttMap(BaseConfig):
    def __init__(self, key: str, hist_seq: str):
        self.key = key
        self.hist_seq = hist_seq

    @staticmethod
    def handle(data):
        res = SeqAttMap(data["key"], data["hist_seq"])
        return res


class FeatureGroup(BaseConfig):
    def __init__(self, group_name: str, feature_names: List[str] = None,
                 seq_att_map_list: List[SeqAttMap] = None, seq_att_projection_dim: int = 0,
                 ):
        self.group_name = group_name
        self.feature_names = feature_names

        self.seq_att_map_list = seq_att_map_list
        self.seq_att_projection_dim = seq_att_projection_dim

    @staticmethod
    def handle(data):
        res = FeatureGroup(data["group_name"])
        if "feature_names" in data:
            res.feature_names = data["feature_names"]
        if "seq_att_map_list" in data:
            seq_att_map_list = []
            for seq_att_map in data["seq_att_map_list"]:
                seq_att_map_list.append(SeqAttMap.handle(seq_att_map))
            res.seq_att_map_list = seq_att_map_list
        if "seq_att_projection_dim" in data:
            res.seq_att_projection_dim = data["seq_att_projection_dim"]
        return res

    @property
    def feature_name_list(self):
        if self.feature_names:
            return self.feature_names
        elif self.seq_att_map_list:
            res = []
            for att_map in self.seq_att_map_list:
                if att_map.key and att_map.key not in res:
                    res.append(att_map.key)
                if att_map.hist_seq and att_map.hist_seq not in res:
                    res.append(att_map.hist_seq)
            return res
        else:
            return []


class ConstantLearningRate(BaseConfig):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    @staticmethod
    def handle(data):
        res = ConstantLearningRate()
        if "learning_rate" in data:
            res.learning_rate = data["learning_rate"]
        return res


class ExponentialDecayLearningRate(BaseConfig):
    def __init__(self, initial_learning_rate=0.01, decay_steps=20000, decay_factor=0.95, min_learning_rate=0.0001):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.min_learning_rate = min_learning_rate

    @staticmethod
    def handle(data):
        res = ExponentialDecayLearningRate()
        if "initial_learning_rate" in data:
            res.initial_learning_rate = data["initial_learning_rate"]
        if "decay_steps" in data:
            res.decay_steps = data["decay_steps"]
        if "decay_factor" in data:
            res.decay_factor = data["decay_factor"]
        if "min_learning_rate" in data:
            res.min_learning_rate = data["min_learning_rate"]
        return res


class LearningRate(BaseConfig):
    def __init__(self, learning_rate_type="exponential_decay_learning_rate",
                 constant_learning_rate=ConstantLearningRate(),
                 exponential_decay_learning_rate=ExponentialDecayLearningRate()
                 ):
        self.learning_rate_type = learning_rate_type
        self.constant_learning_rate = constant_learning_rate
        self.exponential_decay_learning_rate = exponential_decay_learning_rate

    @staticmethod
    def handle(data):
        res = LearningRate()
        if "learning_rate_type" in data:
            res.learning_rate_type = data["learning_rate_type"]
        if "constant_learning_rate" in data:
            res.constant_learning_rate = ConstantLearningRate.handle(data["constant_learning_rate"])
        if "exponential_decay_learning_rate" in data:
            res.exponential_decay_learning_rate = ExponentialDecayLearningRate.handle(
                data["exponential_decay_learning_rate"])
        return res


class SgdOptimizer(BaseConfig):
    def __init__(self, learning_rate=LearningRate()):
        self.learning_rate = learning_rate

    @staticmethod
    def handle(data):
        res = SgdOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        return res


class AdagradOptimizer(BaseConfig):
    def __init__(self, learning_rate=LearningRate()):
        self.learning_rate = learning_rate

    @staticmethod
    def handle(data):
        res = AdagradOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        return res


class AdamOptimizer(BaseConfig):
    def __init__(self, learning_rate=LearningRate(), beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    @staticmethod
    def handle(data):
        res = AdamOptimizer()
        if "learning_rate" in data:
            res.learning_rate = LearningRate.handle(data["learning_rate"])
        if "beta1" in data:
            res.beta1 = data["beta1"]
        if "beta2" in data:
            res.beta2 = data["beta2"]
        return res


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

    @staticmethod
    def handle(data):
        res = Optimizer()
        if "optimizer_type" in data:
            res.optimizer_type = data["optimizer_type"]
        if "sgd_optimizer" in data:
            res.sgd_optimizer = SgdOptimizer.handle(data["sgd_optimizer"])
        if "adagrad_optimizer" in data:
            res.adagrad_optimizer = AdagradOptimizer.handle(data["adagrad_optimizer"])
        if "adam_optimizer" in data:
            res.adam_optimizer = AdamOptimizer.handle(data["adam_optimizer"])
        return res


class TrainConfig(BaseConfig):
    def __init__(self, optimizer_config: Optimizer,
                 log_step_count_steps=1000,
                 save_checkpoints_steps=20000,
                 keep_checkpoint_max=3,
                 train_distribute=None,
                 TF_CONFIG=None,
                 ):
        self.optimizer_config = optimizer_config
        self.log_step_count_steps = log_step_count_steps
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.train_distribute = train_distribute
        self.TF_CONFIG = TF_CONFIG

    @staticmethod
    def handle(data):
        res = TrainConfig(Optimizer.handle(data["optimizer_config"]))
        if "log_step_count_steps" in data:
            res.log_step_count_steps = data["log_step_count_steps"]
        if "save_checkpoints_steps" in data:
            res.save_checkpoints_steps = data["save_checkpoints_steps"]
        if "keep_checkpoint_max" in data:
            res.keep_checkpoint_max = data["keep_checkpoint_max"]
        if "train_distribute" in data:
            res.train_distribute = data["train_distribute"]
        if "TF_CONFIG" in data:
            res.TF_CONFIG = data["TF_CONFIG"]
        return res


class EvalMetric(BaseConfig):
    def __init__(self, name):
        self.name = name


class AUC(EvalMetric):
    def __init__(self):
        super(AUC, self).__init__("auc")

    @staticmethod
    def handle(data):
        res = AUC()
        return res


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

    @staticmethod
    def handle(data):
        res = GroupAUC(data["gid_field"])
        if "reduction" in data:
            res.reduction = data["reduction"]
        return res


class PCOPC(EvalMetric):
    def __init__(self):
        super(PCOPC, self).__init__("pcopc")

    @staticmethod
    def handle(data):
        res = PCOPC()
        return res


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

    @staticmethod
    def handle(data):
        if "auc" in data:
            auc = AUC.handle(data["auc"])
        else:
            auc = None
        if "gauc" in data:
            gauc = GroupAUC.handle(data["gauc"])
        else:
            gauc = None
        if "pcopc" in data:
            pcopc = PCOPC.handle(data["pcopc"])
        else:
            pcopc = None
        res = EvalConfig(auc, gauc, pcopc)
        return res


class ExportConfig(BaseConfig):
    def __init__(self, export_dir):
        self.export_dir = export_dir

    @staticmethod
    def handle(data):
        res = ExportConfig(data["export_dir"])
        return res


class InteractionTower(BaseConfig):
    def __init__(self, input_group: str, interaction_config: InteractionConfig):
        self.input_group = input_group
        self.interaction_config = interaction_config

    @staticmethod
    def handle(data):
        interaction_config = InteractionConfig.handle(data["interaction_config"])
        res = InteractionTower(data["input_group"], interaction_config)
        return res


class MMoEModelCofing(BaseConfig):
    def __init__(self, label_names: List[str], num_expert: int, expert_dnn_config: DNNConfig):
        self.label_names = label_names
        self.num_task = len(label_names)
        self.num_expert = num_expert
        self.expert_dnn_config = expert_dnn_config

    @staticmethod
    def handle(data):
        expert_dnn_config = DNNConfig.handle(data["expert_dnn_config"])
        res = MMoEModelCofing(data["label_names"], data["num_expert"], expert_dnn_config)
        return res


class PLEModelCofing(BaseConfig):
    def __init__(self, label_names: List[str], expert_dnn_config: DNNConfig,
                 num_expert_share: int, num_expert_per_task: int, num_extraction_network: int):
        self.label_names = label_names
        self.num_task = len(label_names)
        self.expert_dnn_config = expert_dnn_config
        self.num_expert_share = num_expert_share
        self.num_expert_per_task = num_expert_per_task
        self.num_extraction_network = num_extraction_network

    @staticmethod
    def handle(data):
        expert_dnn_config = DNNConfig.handle(data["expert_dnn_config"])
        res = PLEModelCofing(data["label_names"], expert_dnn_config,
                             data["num_expert_share"], data["num_expert_per_task"], data["num_extraction_network"])
        return res


class BiasTower(BaseConfig):
    def __init__(self, input_group):
        self.input_group = input_group

    @staticmethod
    def handle(data):
        res = BiasTower(data["input_group"])
        return res


class ModelConfig(BaseConfig):
    def __init__(self, model_class: str,
                 feature_groups: List[FeatureGroup],

                 esmm_model_config: ESMMModelConfig = None,
                 esmm_v2_model_config: ESMMV2ModelConfig = None,
                 aitm_model_config: AITMModelConfig = None,
                 mmoe_model_config: MMoEModelCofing = None,
                 ple_model_config: PLEModelCofing = None,

                 wide_towers: List[str] = None,
                 dnn_towers: List[DNNTower] = None,
                 interaction_towers: List[InteractionTower] = None,
                 din_towers: List[DINTower] = None,
                 bst_towers: List[BSTTower] = None,
                 dien_towers: List[DIENTower] = None,

                 final_dnn: DNNConfig = None,
                 bias_tower: BiasTower = None,

                 embedding_regularization: float = 0.0,
                 l2_regularization: float = 0.0001,
                 use_dynamic_embedding: bool = False,
                 pretrain_variable_dir: str = None,
                 ):
        self.model_class = model_class
        self.feature_groups = feature_groups

        self.esmm_model_config = esmm_model_config
        self.esmm_v2_model_config = esmm_v2_model_config
        self.aitm_model_config = aitm_model_config
        self.mmoe_model_config = mmoe_model_config
        self.ple_model_config = ple_model_config

        self.wide_towers = wide_towers
        self.dnn_towers = dnn_towers
        self.interaction_towers = interaction_towers
        self.din_towers = din_towers
        self.bst_towers = bst_towers
        self.dien_towers = dien_towers

        self.final_dnn = final_dnn
        self.bias_tower = bias_tower

        self.embedding_regularization = embedding_regularization
        self.l2_regularization = l2_regularization

        self.use_dynamic_embedding = use_dynamic_embedding
        self.pretrain_variable_dir = pretrain_variable_dir

    @staticmethod
    def handle(data):
        feature_groups = []
        for feature_group in data["feature_groups"]:
            feature_groups.append(FeatureGroup.handle(feature_group))
        res = ModelConfig(data["model_class"], feature_groups)

        if "esmm_model_config" in data:
            res.esmm_model_config = ESMMModelConfig.handle(data["esmm_model_config"])
        if "esmm_v2_model_config" in data:
            res.esmm_v2_model_config = ESMMV2ModelConfig.handle(data["esmm_v2_model_config"])
        if "aitm_model_config" in data:
            res.aitm_model_config = AITMModelConfig.handle(data["aitm_model_config"])
        if "mmoe_model_config" in data:
            res.mmoe_model_config = MMoEModelCofing.handle(data["mmoe_model_config"])
        if "ple_model_config" in data:
            res.ple_model_config = PLEModelCofing.handle(data["ple_model_config"])

        if "wide_towers" in data:
            res.wide_towers = data["wide_towers"]
        if "dnn_towers" in data:
            dnn_towers = []
            for tower in data["dnn_towers"]:
                dnn_towers.append(DNNTower.handle(tower))
            res.dnn_towers = dnn_towers
        if "interaction_towers" in data:
            interaction_towers = []
            for tower in data["interaction_towers"]:
                interaction_towers.append(InteractionTower.handle(tower))
            res.interaction_towers = interaction_towers
        if "din_towers" in data:
            din_towers = []
            for tower in data["din_towers"]:
                din_towers.append(DINTower.handle(tower))
            res.din_towers = din_towers
        if "bst_towers" in data:
            bst_towers = []
            for tower in data["bst_towers"]:
                bst_towers.append(BSTTower.handle(tower))
            res.bst_towers = bst_towers
        if "dien_towers" in data:
            dien_towers = []
            for tower in data["dien_towers"]:
                dien_towers.append(DIENTower.handle(tower))
            res.dien_towers = dien_towers

        if "final_dnn" in data:
            res.final_dnn = DNNConfig.handle(data["final_dnn"])
        if "bias_tower" in data:
            res.bias_tower = BiasTower.handle(data["bias_tower"])

        if "embedding_regularization" in data:
            res.embedding_regularization = data["embedding_regularization"]
        if "l2_regularization" in data:
            res.l2_regularization = data["l2_regularization"]

        if "use_dynamic_embedding" in data:
            res.use_dynamic_embedding = data["use_dynamic_embedding"]

        if "pretrain_variable_dir" in data:
            res.pretrain_variable_dir = data["pretrain_variable_dir"]
        return res


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
        res = PipelineConfig(data["model_dir"],
                             InputConfig.handle(data["input_config"]),
                             FeatureConfig.handle(data["feature_config"]),
                             ModelConfig.handle(data["model_config"]),
                             )
        if "train_config" in data:
            res.train_config = TrainConfig.handle(data["train_config"])
        if "eval_config" in data:
            res.eval_config = EvalConfig.handle(data["eval_config"])
        if "export_config" in data:
            res.export_config = ExportConfig.handle(data["export_config"])
        return res
