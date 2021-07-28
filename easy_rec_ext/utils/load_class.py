# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/7/23 5:47 下午
# desc:

import logging
import pydoc


def load_by_path(path):
    """Load functions or modules or classes.

    Args:
      path: path to modules or functions or classes,
          such as: tf.nn.relu

    Return:
      modules or functions or classes
    """
    path = path.strip()
    if path == "" or path is None:
        return None
    components = path.split(".")
    if components[0] == "tf":
        components[0] = "tensorflow"
    path = ".".join(components)
    try:
        return pydoc.locate(path)
    except pydoc.ErrorDuringImport:
        logging.error("load %s failed" % path)
        return None
