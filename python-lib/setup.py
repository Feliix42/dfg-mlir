from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dfg-mlir-tools",
    version="0.1.0",
    author="Jiahong Bi",
    author_email="jiahong.bi@tu-dresden.de",
    description="Tools for working with dfg-mlir in Python",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "tensorflow<=2.18.0",
        "numpy<=2.0.2",
        "iree-base-compiler<=3.1.0",
        "iree-base-runtime<=3.1.0",
        "iree-tools-tflite<=20250107.1133"
    ],
)