# DFG MLIR Tools

A collection of tools for working with dfg-mlir in Python.

## Installation

```bash
# Install the package
pip install .
```

## Usage

After installation, you can import modules from the package:

```python
# Import the TFLite to JSON converter
from lib.ml_model.get_layer_info.tflite_to_json import convert_tflite_to_json

# Extract layer infos from a TFLite model to JSON
convert_tflite_to_json("model.tflite", "output.json")
```

## Features

- **TFLite to JSON Converter**: Convert TensorFlow Lite models to JSON format for retrieving network layers in dfg-mlir transformation pass.

## Requirements

- Python 3.11
- TensorFlow <= 2.18.0
- NumPy 2.0.2
- iree-base-compiler <= 3.1.0
- iree-base-runtime <= 3.1.0
- iree-tools-tflite <= 20250107.1133