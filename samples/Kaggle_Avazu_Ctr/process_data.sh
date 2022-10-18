#!/bin/bash

if [ -z ${easy_rec_ext_data_dir} ]; then
  echo "easy_rec_ext_data_dir is not exists"
  exit 1
else
  echo "easy_rec_ext_data_dir=${easy_rec_ext_data_dir}"
fi

for part in 21 22 23 24 25 26 27 28 29 30
do
cat ${easy_rec_ext_data_dir}/train | grep ",1410${part}" > ${easy_rec_ext_data_dir}/sample_${part}
done
