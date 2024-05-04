# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="segment_anything",
    version="1.0",
    install_requires=['efficientnet_pytorch'],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime", "timm"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
