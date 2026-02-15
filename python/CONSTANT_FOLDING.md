# Constant Folding and Host/Device Routing

This document describes the constant folding and host/device routing features added to the LuisaCompute Python DSL v2.

## Overview

The DSL now has the ability to:

1. **Constant-fold values on the fly**: Math operations on constants are evaluated at compile time
2. **Correctly route builtin calls**: Automatically decides whether to compute on host (for constants) or emit device instructions (for DSL values)

## The `@router` Decorator

The `@router` decorator is the core mechanism for enabling both constant folding and host/device routing.

### Usage

```python
from luisa import router
from luisa.lang.ir import Op
import math

# Simple usage with a host implementation and device op
@router(host_impl=math.sin, device_op=Op.SIN)
def sin(x):
    pass

# Usage with a custom device wrapper (for functions with special handling)
def _lerp_device_wrapper(builder, a, b, t):
    return builder._emit(Op.LERP, a.type, [a, b, t])

@router(host_impl=lambda a, b, t: a + (b - a) * t,
        device_wrapper=_lerp_device_wrapper)
def lerp(a, b, t):
    pass
```

### How It Works

When a routed function is called:

1. **Check if all arguments are constants**: Uses `is_constant_value()` to check each argument
2. **If all are constants**: Extract Python values, call the host implementation, wrap result in `ConstantValue`
3. **If any is a DSL value**: Convert all args to IR values and emit device instruction

### Example

```python
from luisa import sin, Float, kernel, Buffer

# Constant folding: sin(1.0 + 2.0) creates ConstantValue of sin(3.0)
a = sin(1.0 + 2.0)  # => ConstantValue(0.14112...)

# Device routing: sin(x) where x is a DSL value
@kernel
def my_kernel(buf: Buffer[Float], x: Float):
    buf[0] = sin(x)  # => Emits device-side SIN instruction
```

## Math Functions with Constant Folding

The following math functions now support constant folding:

### Unary Functions
- `sqrt`, `abs`, `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`
- `exp`, `exp2`, `log`, `log2`, `log10`
- `floor`, `ceil`, `round`, `trunc`, `fract`, `saturate`

### Binary Functions
- `min`, `max`, `clamp`, `lerp`, `step`, `smoothstep`, `pow`, `atan2`

### Vector Functions (Device-only)
- `normalize`, `length`, `length_squared`, `dot`, `cross`, `distance`
- `reflect`, `refract`, `faceforward`

### Matrix Functions (Device-only)
- `transpose`, `inverse`, `determinant`

## ConstantValue Arithmetic

`ConstantValue` now supports arithmetic operations, allowing expressions like:

```python
from luisa import sin, cos

# All operations produce ConstantValue results
result = sin(0.5) * cos(0.25) + sqrt(0.5)
```

Supported operations:
- `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Unary `-`, `+`, `abs()`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`

## API Reference

### Router Functions

```python
from luisa import (
    router,              # Decorator for creating routed functions
    RoutedFunction,      # Class returned by @router
    is_constant_value,   # Check if a value is a compile-time constant
    extract_constant_value,  # Extract Python value from constant
    VectorValue,         # Compile-time vector value
)
```

### `router` Decorator

```python
def router(
    host_impl: Optional[Callable] = None,      # Python function for constant folding
    device_op: Optional[Op] = None,            # IR operation for device execution
    device_wrapper: Optional[Callable] = None  # Custom device emission function
)
```

### `is_constant_value(val)`

Returns `True` if the value is a compile-time constant (Python primitive or `ConstantValue`).

```python
from luisa import is_constant_value, ConstantValue, Float

is_constant_value(1.0)           # True
is_constant_value(True)          # True
is_constant_value(None)          # True
is_constant_value(ConstantValue(typ=Float, value=3.14))  # True
```

## Benefits

1. **Performance**: Constant expressions are evaluated at compile time, reducing runtime overhead
2. **Simplicity**: Same code works for both constants and DSL values - no manual optimization needed
3. **Composability**: Complex expressions with mixed constants and DSL values work seamlessly

## Example: Optimized Kernel

```python
from luisa import kernel, Buffer, Float, sin, lerp

@kernel
def optimized_kernel(buf: Buffer[Float]):
    # Constants are folded
    PI = 3.14159265359
    TWO_PI = 2.0 * PI  # Folded to 6.28318...
    
    idx = 0
    
    # sin(TWO_PI) is computed at compile time (~0.0)
    offset = sin(TWO_PI)
    
    # This becomes just: buf[0] = 0.0 + 1.0 = 1.0
    buf[idx] = offset + 1.0
```

Generated IR (optimized):
```llvm
kernel void optimized_kernel(buffer<f32> arg0) {
  buffer_write(arg0, 0, 1.0);
}
```

Without constant folding, this would generate multiple instructions for the computation.

## Testing

Run the constant folding tests:

```bash
cd python
pytest tests/test_constant_folding.py -v
```

Run the demo:

```bash
python examples/constant_folding_demo.py
```
