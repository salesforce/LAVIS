#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from setuptools import find_packages, setup, find_namespace_packages


setup(
    name="lavis",
    version="0.0.1",
    author="Dongxu Li, Junnan Li, Hung Le, Steven C.H. Hoi",
    description="LAVIS - An Extensible Library of Language-Vision Models and Datasets",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="LAVIS - An Extensible Library of Language-Vision Models and Datasets",
    # url="https://github.com/salesforce/omnixai",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="lavis.*"),
    install_requires=[
        "omegaconf>=2.1.2",
        "opencv-python>=4.5.5",
        "pycocoevalcap",
        "pycocotools",
        "timm==0.4.12",
        "torch==1.10.0",
        "torchvision==0.11.1",
        "fairscale==0.4.4",
        "transformers==4.15.0",
        "einops==0.4.1",
        "decord>=0.6.0",
        "tqdm",
        "wheel",
        "packaging",
        "ipython",
    ],
    python_requires=">=3.7.0",
    include_package_data=True,
    zip_safe=False,
)
