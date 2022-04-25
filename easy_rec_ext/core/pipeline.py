# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/25 9:37 下午
# desc:

from typing import List
from easy_rec_ext.builders.optimizer_builder import Optimizer

from easy_rec_ext.layers.dnn import DNNConfig, DNNTower
from easy_rec_ext.layers.sequence_pooling import SequencePoolingConfig, SequencePoolingTower
from easy_rec_ext.layers.interaction import InteractionConfig

from easy_rec_ext.model.dssm import DSSMModelConfig
from easy_rec_ext.model.dropoutnet import DropoutNetModelConfig
from easy_rec_ext.model.sdm import SDMModelConfig
from easy_rec_ext.model.mind import MINDModelConfig

from easy_rec_ext.model.din import DINTower
from easy_rec_ext.model.bst import BSTTower
from easy_rec_ext.model.dien import DIENTower
from easy_rec_ext.model.can import CANTower

from easy_rec_ext.model.mmoe import MMoEModelCofing
from easy_rec_ext.model.esmm import ESMMModelConfig
from easy_rec_ext.model.aitm import AITMModelConfig

from easy_rec_ext.model.star import STARModelConfig


class BaseConfig(object):
    def __str__(self):
        return str(self.__dict__)


class InputField(BaseConfig):
    def __init__(self, input_name: str, input_type: str):
        self.input_name = input_name
        self.input_type = input_type

    @staticmethod
    def handle(data):
        input_name = data["input_name"]
        input_type = data["input_type"]
        assert input_name, "input_name must not be empty"
        assert input_type in ["int", "float", "string"], "input_type must in [int,float,string]"
        res = InputField(input_name, input_type)
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
                 input_name: str, feature_type: str, feature_name: str = None,
                 raw_input_dim: int = 1, raw_input_embedding_type: str = None,
                 one_hot: int = 0,
                 embedding_name: str = None, embedding_dim: int = 16,
                 num_buckets: int = 0, hash_bucket_size: int = 0,
                 sequence_pooling_config: SequencePoolingConfig = None,
                 limit_seq_size: int = None,
                 ):
        self.input_name = input_name
        self.feature_type = feature_type
        self.feature_name = feature_name if feature_name else input_name

        self.raw_input_dim = raw_input_dim
        self.raw_input_embedding_type = raw_input_embedding_type

        self.one_hot = one_hot

        self.embedding_name = embedding_name if embedding_name else input_name + "_embedding"
        self.embedding_dim = embedding_dim

        self.num_buckets = num_buckets
        self.hash_bucket_size = hash_bucket_size

        self.sequence_pooling_config = sequence_pooling_config
        self.limit_seq_size = limit_seq_size

    @staticmethod
    def handle(data):
        feature_type = data["feature_type"]
        assert feature_type in ["AuxiliaryFeature", "IdFeature", "RawFeature", "SequenceFeature"]

        res = FeatureField(data["input_name"], feature_type)

        if "feature_name" in data:
            res.feature_name = data["feature_name"] if data["feature_name"] else res.input_name

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
        if "limit_seq_size" in data:
            res.limit_seq_size = data["limit_seq_size"]
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


class CartesianInteractionMap(BaseConfig):
    def __init__(self, user_keys: List[str], item_key: str):
        self.user_keys = user_keys
        self.item_key = item_key

    @staticmethod
    def handle(data):
        user_keys = data["user_keys"]
        item_key = data["item_key"]
        assert user_keys
        assert item_key
        assert item_key not in user_keys
        res = CartesianInteractionMap(user_keys, item_key)
        return res


class SENetLayerConfig(BaseConfig):
    def __init__(self, reduction_ratio=2):
        self.reduction_ratio = reduction_ratio

    @staticmethod
    def handle(data):
        res = SENetLayerConfig()
        if "reduction_ratio" in data:
            res.reduction_ratio = data["reduction_ratio"]
        return res


class FeatureGroup(BaseConfig):
    def __init__(self, group_name: str, feature_names: List[str] = None,
                 seq_att_map_list: List[SeqAttMap] = None, seq_att_projection_dim: int = 0,
                 cartesian_interaction_map: CartesianInteractionMap = None,
                 senet_layer_config: SENetLayerConfig = None,
                 ):
        self.group_name = group_name
        self.feature_names = feature_names

        self.seq_att_map_list = seq_att_map_list
        self.seq_att_projection_dim = seq_att_projection_dim

        self.cartesian_interaction_map = cartesian_interaction_map

        self.senet_layer_config = senet_layer_config

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

        if "cartesian_interaction_map" in data:
            res.cartesian_interaction_map = CartesianInteractionMap.handle(data["cartesian_interaction_map"])

        if "senet_layer_config" in data:
            res.senet_layer_config = SENetLayerConfig.handle(data["senet_layer_config"])
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
    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def handle(data):
        res = EvalMetric(data["name"])
        return res

    def __str__(self):
        return "EvalMetric[%s]" % self.name


class AUC(EvalMetric):
    def __init__(self):
        super(AUC, self).__init__("auc")

    @staticmethod
    def handle(data):
        res = AUC()
        return res


class GroupAUC(EvalMetric):
    def __init__(self, gid_field: str, reduction: str = "mean_by_sample_num"):
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

    def __str__(self):
        return "EvalMetric[gauc@%s]" % self.gid_field


class PCOPC(EvalMetric):
    def __init__(self):
        super(PCOPC, self).__init__("pcopc")

    @staticmethod
    def handle(data):
        res = PCOPC()
        return res


class RecallAtK(EvalMetric):
    def __init__(self, topk: int = 100):
        super(RecallAtK, self).__init__("recall_at_k")
        self.topk = topk

    @staticmethod
    def handle(data):
        res = RecallAtK()
        if "topk" in data:
            res.topk = data["topk"]
        return res

    def __str__(self):
        return "EvalMetric[recall@%d]" % self.topk


class PrecisionAtK(EvalMetric):
    def __init__(self, topk: int = 100):
        super(PrecisionAtK, self).__init__("precision_at_k")
        self.topk = topk

    @staticmethod
    def handle(data):
        res = PrecisionAtK()
        if "topk" in data:
            res.topk = data["topk"]
        return res

    def __str__(self):
        return "EvalMetric[precision@%d]" % self.topk


class EvalConfig(BaseConfig):
    def __init__(self,
                 metric_set,
                 ):
        self.metric_set = metric_set

    @staticmethod
    def handle(data):
        metric_set = []
        for metric in data["metric_set"]:
            eval_metric = EvalMetric.handle(metric)
            if eval_metric.name == "auc":
                metric_set.append(AUC.handle(metric))
            elif eval_metric.name == "gauc":
                metric_set.append(GroupAUC.handle(metric))
            elif eval_metric.name == "pcopc":
                metric_set.append(PCOPC.handle(metric))
            elif eval_metric.name == "recall_at_k":
                metric_set.append(RecallAtK.handle(metric))
            elif eval_metric.name == "precision_at_k":
                metric_set.append(PrecisionAtK.handle(metric))
            else:
                raise NotImplemented
        res = EvalConfig(metric_set=metric_set)
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
    def __init__(self, input_group, fusion_mode="add"):
        self.input_group = input_group
        self.fusion_mode = fusion_mode

    @staticmethod
    def handle(data):
        res = BiasTower(data["input_group"])
        if "fusion_mode" in data:
            assert data["fusion_mode"] in ["add", "multiply"]
            res.fusion_mode = data["fusion_mode"]
        return res


class ModelConfig(BaseConfig):
    def __init__(self, model_class: str,
                 feature_groups: List[FeatureGroup],

                 # Match Model Config
                 dssm_model_config: DSSMModelConfig = None,
                 dropoutnet_model_config: DropoutNetModelConfig = None,
                 sdm_model_config: SDMModelConfig = None,
                 mind_model_config: MINDModelConfig = None,

                 # Multi-Task Rank Model Config
                 esmm_model_config: ESMMModelConfig = None,
                 aitm_model_config: AITMModelConfig = None,
                 mmoe_model_config: MMoEModelCofing = None,
                 ple_model_config: PLEModelCofing = None,

                 wide_towers: List[str] = None,
                 dnn_towers: List[DNNTower] = None,
                 seq_pooling_towers: List[SequencePoolingTower] = None,
                 interaction_towers: List[InteractionTower] = None,
                 din_towers: List[DINTower] = None,
                 bst_towers: List[BSTTower] = None,
                 dien_towers: List[DIENTower] = None,
                 can_towers: List[CANTower] = None,

                 final_dnn: DNNConfig = None,
                 bias_towers: List[BiasTower] = None,
                 star_model_config: STARModelConfig = None,

                 embedding_regularization: float = 0.0,
                 l2_regularization: float = 0.0,
                 use_dynamic_embedding: bool = False,
                 pretrain_variable_dir: str = None,
                 ):
        self.model_class = model_class
        self.feature_groups = feature_groups

        self.dssm_model_config = dssm_model_config
        self.dropoutnet_model_config = dropoutnet_model_config
        self.sdm_model_config = sdm_model_config
        self.mind_model_config = mind_model_config

        self.esmm_model_config = esmm_model_config
        self.aitm_model_config = aitm_model_config
        self.mmoe_model_config = mmoe_model_config
        self.ple_model_config = ple_model_config

        self.wide_towers = wide_towers
        self.dnn_towers = dnn_towers
        self.seq_pooling_towers = seq_pooling_towers
        self.interaction_towers = interaction_towers
        self.din_towers = din_towers
        self.bst_towers = bst_towers
        self.dien_towers = dien_towers
        self.can_towers = can_towers

        self.final_dnn = final_dnn
        self.bias_towers = bias_towers
        self.star_model_config = star_model_config

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

        if "dssm_model_config" in data:
            res.dssm_model_config = DSSMModelConfig.handle(data["dssm_model_config"])
        if "dropoutnet_model_config" in data:
            res.dropoutnet_model_config = DropoutNetModelConfig.handle(data["dropoutnet_model_config"])
        if "sdm_model_config" in data:
            res.sdm_model_config = SDMModelConfig.handle(data["sdm_model_config"])
        if "mind_model_config" in data:
            res.mind_model_config = MINDModelConfig.handle(data["mind_model_config"])

        if "esmm_model_config" in data:
            res.esmm_model_config = ESMMModelConfig.handle(data["esmm_model_config"])
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
        if "seq_pooling_towers" in data:
            seq_pooling_towers = []
            for tower in data["seq_pooling_towers"]:
                seq_pooling_towers.append(SequencePoolingTower.handle(tower))
            res.seq_pooling_towers = seq_pooling_towers
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
        if "can_towers" in data:
            can_towers = []
            for tower in data["can_towers"]:
                can_towers.append(CANTower.handle(tower))
            res.can_towers = can_towers

        if "final_dnn" in data:
            res.final_dnn = DNNConfig.handle(data["final_dnn"])
        if "bias_towers" in data:
            bias_towers = []
            for tower in data["bias_towers"]:
                bias_towers.append(BiasTower.handle(tower))
            res.bias_towers = bias_towers
        if "star_model_config" in data:
            res.star_model_config = STARModelConfig.handle(data["star_model_config"])

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
    def __init__(self,
                 model_dir: str,
                 input_config: InputConfig,
                 feature_config: FeatureConfig,
                 model_config: ModelConfig,
                 train_config: TrainConfig = None,
                 eval_config: EvalConfig = None,
                 export_config: ExportConfig = None,
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
