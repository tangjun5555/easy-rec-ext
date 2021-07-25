# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:36 下午
# desc:

from pathlib import Path
from setuptools import setup, find_packages

version = "0.0.1"


setup(
    name="easy_rec_ext",
    version=version,
    description="An TensorFlow Framework For Recommender System",
    packages=find_packages(),
    author="tangj",
    author_email="1844250138@qq.com",
    python_requires=">=3.5",
    install_requires=Path("requirements.txt").read_text().splitlines(),
    extras_require={
        "cpu": ["tensorflow>=1.13.1"],
        "gpu": ["tensorflow-gpu>=1.13.1"],
    },
)
