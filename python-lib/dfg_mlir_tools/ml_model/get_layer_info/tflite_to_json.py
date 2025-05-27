"""
TFLite to JSON Converter Library

This library provides functionality to convert TensorFlow Lite models into a JSON 
representation of their layer structure, which can be useful for analysis or visualization.

Usage:
    from python.lib.ml_model.get_layer_info.tflite_to_json import convert_tflite_to_json
    
    # Basic usage
    convert_tflite_to_json("model.tflite", "model.json")
"""

import json
import os
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.python.schema_py_generated import BuiltinOperator


def get_tensor_shape(tensor):
    """
    Get the shape of a tensor as a list.
    
    Args:
        tensor: TFLite tensor object
        
    Returns:
        List of dimensions or empty list if shape is None
    """
    shape = tensor.ShapeAsNumpy()
    return shape.tolist() if shape is not None else []


def get_tensor_name(tensor):
    """
    Get the name of a tensor as a UTF-8 string.
    
    Args:
        tensor: TFLite tensor object
        
    Returns:
        String representation of tensor name
    """
    return tensor.Name().decode("utf-8")


def get_tensor_type(tensor):
    """
    Get the data type of a tensor as a readable string.
    
    Args:
        tensor: TFLite tensor object
        
    Returns:
        String representation of tensor type (e.g., "f32" for FLOAT32)
    """
    tensor_type = tensor.Type()
    type_mapping = {
        schema.TensorType.FLOAT32: "f32",
        schema.TensorType.FLOAT16: "f16",
        schema.TensorType.INT8: "i8",
        schema.TensorType.UINT8: "ui8",
        schema.TensorType.INT16: "i16",
        schema.TensorType.UINT16: "ui16",
        schema.TensorType.INT32: "i32",
        schema.TensorType.UINT32: "ui32",
    }
    return type_mapping.get(tensor_type, f"unknown_{tensor_type}")


def load_tflite_model(path):
    """
    Load a TensorFlow Lite model from a file.
    
    Args:
        path: Path to the TFLite model file
        
    Returns:
        Loaded TFLite model object
        
    Raises:
        Exception: If there is an error loading the model
    """
    try:
        with open(path, "rb") as f:
            buf = f.read()
        return schema.Model.GetRootAsModel(buf, 0)
    except Exception as e:
        raise Exception(f"Error loading TFLite model: {e}")


def save_to_json(data, filename):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to be saved as JSON
        filename: Output filename
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def convert_tflite_to_json(model_path, json_path):
    """
    Convert a TFLite model to a JSON representation.
    
    Args:
        model_path: Path to the input TFLite model
        json_path: Path where the JSON output will be saved
        
    Returns:
        List of layer information dictionaries that were saved to the JSON file
        
    Raises:
        Exception: If there is an error during conversion
    """
    try:
        # Check if input file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model file not found: {model_path}")
            
        # Create builtin operator map
        builtin_operator_map = {v: k for k, v in BuiltinOperator.__dict__.items() if not k.startswith("__")}
        
        # Load model
        model = load_tflite_model(model_path)
        subgraph = model.Subgraphs(0)
        operators = subgraph.OperatorsLength()
        tensors = subgraph.Tensors
        operator_codes = model.OperatorCodes
        
        # Track non-constant tensors, model inputs and outputs
        non_constant_tensors = set()
        model_inputs = set()
        model_outputs = set()

        # Identify model inputs
        for i in range(subgraph.InputsLength()):
            model_inputs.add(subgraph.Inputs(i))
            non_constant_tensors.add(subgraph.Inputs(i))

        # Identify model outputs
        for i in range(subgraph.OutputsLength()):
            model_outputs.add(subgraph.Outputs(i))
            non_constant_tensors.add(subgraph.Outputs(i))

        # Identify non-constant tensors (operator outputs)
        for i in range(operators):
            op = subgraph.Operators(i)
            for j in range(op.OutputsLength()):
                non_constant_tensors.add(op.Outputs(j))

        # Initialize data structures for layer information
        layers_info = []
        map_to_new_idx = {}
        real_idx = 0
        tensor_source = {}
        tensor_redirect = {}

        # Process each operator
        for i in range(operators):
            op = subgraph.Operators(i)
            op_code_index = op.OpcodeIndex()
            builtin_code = operator_codes(op_code_index).BuiltinCode()
            op_type = builtin_operator_map.get(builtin_code, f"UNKNOWN_OP_{builtin_code}")

            # Determine start_op based on operator type
            if op_type == "FULLY_CONNECTED":
                start_op = "conv2d"
            elif op_type == "SOFTMAX":
                start_op = "cast"
            elif op_type == "AVERAGE_POOL_2D":
                start_op = "avg_pool2d"
            else:
                start_op = op_type.lower().replace('_', '')
            
            # Skip reshape layers with specific conditions
            skip_layer = False
            if op_type == "RESHAPE":
                input_index = op.Inputs(0)
                input_tensor = tensors(input_index)
                input_shape = get_tensor_shape(input_tensor)
                if all(dim == 1 for dim in input_shape[:-1]):
                    skip_layer = True
                    output_index = op.Outputs(0)
                    tensor_redirect[output_index] = input_index

            if skip_layer:
                continue

            map_to_new_idx[i] = real_idx

            # Process inputs (excluding constants)
            inputs = []
            for j in range(op.InputsLength()):
                input_index = op.Inputs(j)
                if input_index < 0:
                    continue
                if input_index in tensor_redirect:
                    input_index = tensor_redirect[input_index]
                is_const = False
                if input_index not in non_constant_tensors:
                    is_const = True
                    
                input_tensor = tensors(input_index)
                input_shape = get_tensor_shape(input_tensor)
                input_type = get_tensor_type(input_tensor)
                input_name = get_tensor_name(input_tensor)
                
                inputs.append({
                    "shape": input_shape,
                    "type": input_type,
                    "is_const": is_const
                })

            # Process output (each layer should only have one output)
            output_index = op.Outputs(0)
            output_shape = get_tensor_shape(tensors(output_index))
            output_type = get_tensor_type(tensors(output_index))

            tensor_source[output_index] = real_idx

            # Add layer information
            layers_info.append({
                "index": real_idx,
                "name": op_type,
                "start_op": start_op,
                "inputs": inputs,
                "output": [{"shape": output_shape, "type": output_type, "is_const": False}]
            })

            real_idx += 1
            
        # Save results to JSON
        save_to_json(layers_info, json_path)
        
        return layers_info
        
    except Exception as e:
        raise Exception(f"Error converting TFLite to JSON: {e}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TFLite model to JSON representation")
    parser.add_argument("model_path", help="Path to the TFLite model")
    parser.add_argument("json_path", help="Path where the JSON output will be saved")
    
    args = parser.parse_args()
    
    try:
        layers_info = convert_tflite_to_json(args.model_path, args.json_path)
        print(f"Conversion successful! Layers information saved to {args.json_path}")
        print(f"Total layers processed: {len(layers_info)}")
    except Exception as e:
        print(f"Error: {e}")