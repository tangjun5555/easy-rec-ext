##EasyRecExt，对比EasyRec，新增以下特性
###1、使用DynamicEmbedding替换原生TFEmbedding
###2、RankModel增加对BiasFeature的特殊处理
###3、variable支持finetune
###4、支持真正意义上的共享variable
###5、支持自由组合DNN结构，如在一个模型可同时使用FM、DIN、BST等特征交叉结构
###6、支持直接从OSS读取数据集

##TODO
###1、serving阶段对DynamicEmbedding的支持
###2、支持分布式训练