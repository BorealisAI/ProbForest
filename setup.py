# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import codecs
import os.path
import setuptools
from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="prob_forest",
    version=get_version("prob_forest/__init__.py"),
    description="Smooth Approximations of Decision Trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BorealisAI/ProbForest",
    classifiers=[
        'Programming Language :: Python :: 3.7.5',
    ],
    author="Giuseppe Castiglione",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.1',
        'scikit-learn>=0.24.1',
        'xgboost>=1.3.0',
        'lightgbm==3.2.1'
    ],
)
