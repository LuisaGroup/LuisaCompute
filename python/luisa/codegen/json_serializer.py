"""
JSON CodeGen for the LuisaCompute Python DSL v2.

Serializes IR to JSON format for exchange with the C++ backend.
"""

from __future__ import annotations
import json
from typing import Any, TYPE_CHECKING
from dataclasses import asdict, is_dataclass

if TYPE_CHECKING:
    from ..ir import IRFunction, IRModule, IRBasicBlock, IRInstruction, Value
    from ..dsl_types import Type


class IRJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for IR types."""
    
    def default(self, obj: Any) -> Any:
        # Handle dataclasses
        if is_dataclass(obj):
            result = asdict(obj)
            # Add type discriminator
            result['_type'] = obj.__class__.__name__
            return result
        
        # Handle enums
        if hasattr(obj, 'name'):
            return {'_type': obj.__class__.__name__, 'value': obj.name}
        
        # Handle types
        if hasattr(obj, '__dict__'):
            return {
                '_type': obj.__class__.__name__,
                **{k: self.default(v) for k, v in obj.__dict__.items()}
            }
        
        return super().default(obj)


def type_to_dict(t: Type) -> dict[str, Any]:
    """Convert a Type to a dictionary representation."""
    from ..dsl_types import (
        Scalar, Vector, Matrix, Array, Struct, Buffer,
        Texture2D, Texture3D, BindlessArray, Accel, RayQuery, Callable, Void
    )
    
    if isinstance(t, Void):
        return {'kind': 'void'}
    
    if isinstance(t, Scalar):
        return {'kind': 'scalar', 'dtype': t.dtype.name}
    
    if isinstance(t, Vector):
        return {
            'kind': 'vector',
            'element': type_to_dict(t.element),
            'size': t.size
        }
    
    if isinstance(t, Matrix):
        return {
            'kind': 'matrix',
            'element': type_to_dict(t.element),
            'size': t.size
        }
    
    if isinstance(t, Array):
        return {
            'kind': 'array',
            'element': type_to_dict(t.element),
            'size': t.size
        }
    
    if isinstance(t, Struct):
        return {
            'kind': 'struct',
            'name': t.name,
            'fields': [
                {'name': name, 'type': type_to_dict(typ)}
                for name, typ in t.fields
            ],
            'alignment': t.alignment
        }
    
    if isinstance(t, Buffer):
        return {
            'kind': 'buffer',
            'element': type_to_dict(t.element)
        }
    
    if isinstance(t, Texture2D):
        return {
            'kind': 'texture2d',
            'element': type_to_dict(t.element)
        }
    
    if isinstance(t, Texture3D):
        return {
            'kind': 'texture3d',
            'element': type_to_dict(t.element)
        }
    
    if isinstance(t, BindlessArray):
        return {'kind': 'bindless_array'}
    
    if isinstance(t, Accel):
        return {'kind': 'accel'}
    
    if isinstance(t, RayQuery):
        return {'kind': 'ray_query', 'query_any': t.query_any}
    
    if isinstance(t, Callable):
        return {
            'kind': 'callable',
            'arg_types': [type_to_dict(at) for at in t.arg_types],
            'ret_type': type_to_dict(t.ret_type) if t.ret_type else None
        }
    
    return {'kind': 'unknown', 'repr': repr(t)}


def value_to_dict(v: Value) -> dict[str, Any]:
    """Convert a Value to a dictionary representation."""
    from ..ir import ConstantValue, ArgumentValue, InstructionValue
    
    result = {
        'type': type_to_dict(v.type),
    }
    
    if isinstance(v, ConstantValue):
        result['kind'] = 'constant'
        result['value'] = v.value
    
    elif isinstance(v, ArgumentValue):
        result['kind'] = 'argument'
        result['index'] = v.index
    
    elif isinstance(v, InstructionValue):
        result['kind'] = 'instruction'
        result['name'] = v.name
        if v.instruction:
            result['instruction_op'] = v.instruction.op.name
    
    else:
        result['kind'] = 'unknown'
    
    return result


def instruction_to_dict(inst: IRInstruction) -> dict[str, Any]:
    """Convert an IRInstruction to a dictionary."""
    def arg_to_dict(arg):
        if hasattr(arg, 'name'):  # IRBasicBlock
            return {'block': arg.name}
        if hasattr(arg, 'type'):  # Value
            return value_to_dict(arg)
        return arg
    
    return {
        'op': inst.op.name,
        'type': type_to_dict(inst.type),
        'args': [arg_to_dict(a) for a in inst.args],
        'result': inst.result
    }


def basic_block_to_dict(block: IRBasicBlock) -> dict[str, Any]:
    """Convert an IRBasicBlock to a dictionary."""
    return {
        'name': block.name,
        'instructions': [instruction_to_dict(i) for i in block.instructions],
        'is_terminated': block.is_terminated()
    }


def function_to_dict(func: IRFunction) -> dict[str, Any]:
    """Convert an IRFunction to a dictionary."""
    return {
        'name': func.name,
        'arg_types': [type_to_dict(t) for t in func.arg_types],
        'ret_type': type_to_dict(func.ret_type) if func.ret_type else None,
        'is_kernel': func.is_kernel,
        'block_size': func.block_size,
        'blocks': [basic_block_to_dict(b) for b in func.blocks]
    }


def module_to_dict(module: IRModule) -> dict[str, Any]:
    """Convert an IRModule to a dictionary."""
    return {
        'functions': [function_to_dict(f) for f in module.functions]
    }


def serialize_function(func: IRFunction, indent: int | None = 2) -> str:
    """
    Serialize an IRFunction to JSON string.
    
    Args:
        func: The function to serialize
        indent: Indentation level for pretty printing (None for compact)
    
    Returns:
        JSON string representation of the function
    """
    return json.dumps(function_to_dict(func), indent=indent, cls=IRJSONEncoder)


def serialize_module(module: IRModule, indent: int | None = 2) -> str:
    """
    Serialize an IRModule to JSON string.
    
    Args:
        module: The module to serialize
        indent: Indentation level for pretty printing (None for compact)
    
    Returns:
        JSON string representation of the module
    """
    return json.dumps(module_to_dict(module), indent=indent, cls=IRJSONEncoder)


def save_function_to_file(func: IRFunction, path: str, indent: int | None = 2) -> None:
    """Save an IRFunction to a JSON file."""
    with open(path, 'w') as f:
        f.write(serialize_function(func, indent))


def save_module_to_file(module: IRModule, path: str, indent: int | None = 2) -> None:
    """Save an IRModule to a JSON file."""
    with open(path, 'w') as f:
        f.write(serialize_module(module, indent))
