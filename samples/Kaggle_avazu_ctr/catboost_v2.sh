#!/bin/bash

# logloss:0.4080213942	auc:0.7262751619
rm -rf catboost_v2
catboost fit \
  --used-ram-limit=8gb \
  --thread-count=4 \
  --logging-level=Debug \
  --metric-period=1 \
  --train-dir=catboost_v2 \
  --model-format=json \
  --learn-set=train_27_29.csv \
  --test-set=train_30.csv \
  --delimiter=',' \
  --column-description=feature_v2.cd  \
  --loss-function=Logloss \
  --custom-metric="Logloss:hints=skip_train~false,AUC:hints=skip_train~false" \
  --random-seed=555 \
  --iterations=50 \
  --depth=6
