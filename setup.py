# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:36 下午
# desc:

from pathlib import Path
from setuptools import setup, find_packages

version = "0.1.9"

setup(
    name="easy_rec_ext",
    version=version,
    description="An TensorFlow Framework For Recommender System",
    packages=find_packages(),
    author="tangj",
    author_email="1844250138@qq.com",
    python_requires=">=3.6",
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extras_require={
        "cpu": [
            "tensorflow==2.4.1",
            "tensorflow-recommenders-addons==0.2.0"
        ],
        "gpu": [
            "tensorflow-gpu==2.4.1",
            "tensorflow-recommenders-addons-gpu==0.2.0"
        ],
    },
)
