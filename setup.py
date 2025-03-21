#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Core dependencies
core_deps = [
    'pandas',
    'numpy',
    'numba',
]


setup(
    name="fast-twed",
    version="0.0.1",
    author="Alexander Kolpakov, Igor Rivin",
    author_email="akolpakov@uaustin.org, rivin@temple.edu",
    description="A Numba version of TWED",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sashakolpakov/fast-twed",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=core_deps,
)
