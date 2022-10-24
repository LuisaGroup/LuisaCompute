# Python-Luisa User's Guide

Luisa is a Domain-Specific Language(DSL) embedded in Python 
oriented to high performance graphics programming, supporting
Windows, Linux and macOS, with multiple backends supported
including CUDA, DirectX, Metal, LLVM and ISPC.

**This project is still under development. Please [submit an issue](https://github.com/LuisaGroup/LuisaCompute/issues) if you have any problems or suggestions.**

## Compile and Run

```bash
cmake -S . -B build_release -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D LUISA_COMPUTE_ENABLE_PYTHON=ON
cmake --build build_release -j
```

This will produce `luisa` library for Python to include under directory `build_release`.

Note that you may switch `LUISA_COMPUTE_ENABLE_{CUDA|LLVM|ISPC|DX|METAL}` to `ON|OFF` for
different backends, But in order to produce a Python library, `LUISA_COMPUTE_ENABLE_PYTHON` must
be turned `ON`.

> If you build with `LUISA_COMPUTE_ENABLE_CUDA` on, you may face the followig error:
> 
> ```
> CMake Error at src/backends/cuda/CMakeLists.txt:15 (message):
> OptiX_DIR is not defined and OptiX headers are not copied to
> '[Project Directory]/src/backends/cuda/optix'
> ```
> 
> In that case you may either specify the OptiX install directory, or
> manully copy OptiX headers to the given directory.

You can find the artifact at `build_release/bin/lcapi.cpython-<python version>-<platform triple>.so`. 

```bash
cd build_release/bin
python3 test.py
```

This will run the test script and output `mandelbrot.png`.

Currently, we are not yet ready to provide .whl packages, so the best approach to use compiled `luisa`
library is to set `PYTHONPATH` environment variable. For example, you may run

```bash
source set_python_path.sh build_release
```

to add `luisa` to `PYTHONPATH` for your current shell.

## Quick Example

You can see the basic usages of Luisa in the following script. This
program creates 10 threads, each writes 42 into the corresponding position
in the cache.

```python
import luisa

luisa.init('cuda') # backend will be automatically chosen if not specified
b = luisa.Buffer(10, dtype=int)

@luisa.func
def fill(x):
    b.write(dispatch_id().x, x)
    
fill(42, dispatch_size=10) # run `fill` in parallel
print(b.numpy()) # output will be [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
```

## Luisa Functions

Luisa functions are the facilities for mass calculating.

Marking a function with decorator `luisa.func` converts a python function into a luisa one.
Luisa functions follow the same syntax as python functions, although there are minor differences
when invoking them. A luisa function is just-in time(JIT) compiled into statically-typed code 
to run on backend devices when it is called.

A luisa function can be called _in parallel_ in python code. When calling a luisa function
in python, an extra parameter `dispatch_size` representing the number of threads must be given.
For example, in the demo script above when we call `fill(42, dispatch_size=10)`, it creates 10 threads
in parallel, each with the same code but different `dispatch_id`.

A luisa function can also be called within another luisa function. In that case `dispatch_size` is no more needed.

> Python code is host-end code running on CPU, while Luisa code is device-end code running on the acceleration device
> (which could be GPU or CPU itself). Host end creates device-end threads and performs initialization and data collection,
> while device end code is responsible for the major calculation.
> 
> The calculation on device ends are asynchronized by default, meaning that host end can continue executing while device end
> is performing the calculation. Calling `luisa.synchronize()` on host will force the device threads to synchronize,
> and wait until all calculations on device threads are done.
> 
> If you are familiar with CUDA programming, `luisa.func` is similar to `__device__` or `__global__`(depending on where
> it is called).

Luisa functions can have (but not forced) type hints like `def func(a: int)`. If a type hint is provided, the function 
will check whether the given argument is consistent with the type hint. Only single type hint is supported 
(`def func(a: int|float)` is illegal).

> When a Luisa function is called in Python in parallel, the variables passed as arguments are not modified by the code
> inside that function. When calling another Luisa function within a Luisa function, the rules for passing parameters to
> the function are the same as in Python, i.e., scalar types are passed by value, and objects of other types are passed 
> by reference. To avoid ambiguity, direct assignment of values to arguments passed by reference within a function is 
> prohibited. luisa functions can capture external variables, and if they are pure data variables (see the next section),
> they are captured by value at compile time.

Function return values are not supported when calling a Luisa function in Python in parallel. The only way to output a function is to write data to a cache or mapping (see the next section).

The Luisa function currently supports only positional arguments, not keyword arguments, and does not support parameter defaults.

## Types

Unlike Python, Luisa functions are statically typed, i.e. a variable cannot be reassigned to a value of another type 
after it has been defined. We provide the following types:

Scalars, vectors, matrices, arrays, and structures are pure data types whose underlying representation is a fixed-size 
piece of memory in the storage space. Local variables of these types can be created in Luisa functions. Their construction and assignment are copy-by-value, and the rules for passing references are described in the previous section. In general, these types have a uniform way of operating on device side (Luisa functions) and host side (Python code).

Apart from scalars, all objects of pure data types can get a copy on host side by calling `copy`.

Caches, textures, resource indexes, and acceleration structures are resource types that are used to store resources 
shared by all threads. Resources can be referenced in Luisa functions, but creating local variables of resource types 
is not allowed. These types may operate differently on device side (Luisa function) than on host side (Python code). 
Typically, host side is responsible for initializing the data in the resource before the operation starts and collecting 
the data in the resource after the operation ends; the device side uses the elements in the resource for computation.

### Scalar Types

- 32-bit signed integer `int` and unsigned integer `uint`
- single precision floating point `float`
- boolean value `bool`

> `uint` is not well-supported in Luisa yet. You may want to use `int` for now.

Note that the int/float precision in Luisa functions is lower than the int/float precision in python.

### Vector Types

A vector storing a fixed number of scalars, corresponding to a mathematical column vector of up to 4D.

```
luisa.int2, luisa.bool2, luisa.float2,
luisa.int3, luisa.bool3, luisa.float3,
luisa.int4, luisa.bool4, luisa.float4
```

You may import vector and matrix types into the namespace with 

```python
from luisa.mathtypes import *
```

A vector in dimension n can be constructed from 1 or n corresponding scalar(s), for example:

```python
float3(7) # [7.0, 7.0, 7.0]
bool2(True, False) # [True, False]
```

Elements of a vector can be accessed using indices or members.

```python
a[0] == a.x
a[1] == a.y
a[2] == a.z # for vectors with 3 or 4 dimensions
a[3] == a.w # for 4D vectors only
```

All vector types have a class of read-only swizzle members that can be used to form a new vector with 2 to 4 members in 
any order, for example:

```python
int3(7,8,9).xzzy # [7,9,9,8]
float4(1., 2., 3., 4.).zy # [3., 2.]
```

### Matrix Types

A matrix storing a fixed number of scalars, corresponding to a mathematical square matrix of up to 4D.

```python
luisa.float2x2
luisa.float3x3
luisa.float4x4
```

Note that only float matrices are supported.

You may import vector and matrix types into the namespace with

```python
from luisa.mathtypes import *
```

A matrix with dimension n can be constructed from either 1 scalar k (resulting in k times identity matrix) or n times n 
scalars of corresponding types in the order of column precedence. For example:

```python
float2x2(4) # [ 4 0 ]
            # [ 0 4 ]
float2x2(1,2,3,4) # [ 1 3 ]
                  # [ 2 4 ]
                  # printed output will be float2x2([1,2],[3,4])
```

Using indices you can take a column vector from a matrix, readable and writable. For example:

```python
a = float2x2(1, 2, 3, 4)
a[1] = float2(5) # a will be float2x2([1,2],[5,5])
```

### Array Types (`luisa.Array`)

An array can hold a fixed number of elements of the same type, and their element types can be scalars, vectors or matrices.

An array can be constructed from a list, for example:

```python
arr = luisa.array([1, 2, 3, 4, 5])
```

Since arrays are pure data types and can be created as local variables in each thread, the size of arrays is usually 
small (within a few thousand).

Elements of an array can be accessed using indices, e.g. `arr[4]`

### Struct Types (`luisa.Struct`)

A struct can hold a number of members (properties) of a fixed layout, each of which can be of type scalar, vector, matrix, array, or another struct. Structs are used in a similar way to class objects in Python, but because they are pure data types, members cannot be added or removed dynamically.

A structure is constructed using several keyword arguments, for example:

```python
s = luisa.struct(a=42, b=3.14, c=luisa.array([1,2]))
```

You can access the members of a structure using the attribute operator (.) , like `s.a`.

> Note that the order of the members in the storage layout of the structure is the same as the order of the parameters 
> given at the time of construction.
> 
> For example, `luisa.struct(a=42, b=3.14)` is different from `luisa.struct(b=3.14, a=42)`.

Named structure types (see the section _Type Hints_) can have a luisa function as its method (member function).
For example:

```python
struct_t = luisa.StructType(...)

@luisa.func
def f(self, ...):
    ...

struct_t.add_method(f, 'method_name')  # Method name will be function name if not specified
```

This allows instances built from this type to invoke methods in Luisa functions:

```python
a.method_name(...)
```

A constructor (`__init__`) can also be defined. Luisa functions will invoke this method when instantiating this struct.

### Buffer Types (`luisa.buffer`)

A buffer is an array shared by all threads on the device, and its element types can be scalars, vectors, matrices, 
arrays, or structs. Unlike array types, the length of a buffer can be arbitrarily large (limited by the storage space 
of the computing device).

A buffer can be constructed from a luisa array or a numpy array:

```python
buf = luisa.buffer(arr)
```

This operation creates a buffer of the corresponding element type and length on the device and uploads the data. 
If `arr` is a numpy array, its element type can only be `bool`, `numpy.int32` or `numpy.float32`.

You can also create a buffer specifying the type (see the section _Type Hints_) and length of the elements.

```python
buf = luisa.Buffer.empty(size, dtype)  # creates an uninitialized buffer
buf = luisa.Buffer.zeros(size, dtype)  # creates a buffer with all elements initialized as dtype(0)
buf = luisa.Buffer.ones(size, dtype)   # creates a buffer with all elements initialized as dtype(1)
buf = luisa.Buffer.filled(size, value) # creates a buffer with all elements initialized as value
```

You can upload data to an existing buffer from a list or numpy array:

```python
buf.copy_from(arr)  # the data type and length of arr must be identical with buf
```

Access buffered elements directly on host side (in Python code) is not allowed. The buffer can be downloaded to a list, 
or a numpy array:

```python
a1 = buf.to_list()
a2 = buf.numpy()
a3 = numpy.empty(...)
buf.copy_to(a3)
```
> Note that the only corresponding scalar types in numpy are `bool`, `numpy.int32` and `numpy.float32`.
> 
> We recommend using upload/download with numpy arrays _only on scalar buffers_. For non-scalar buffers, we recommend
> using list. If numpy array is used, the user is responsible for the memory layout of the data.

On device side (in Luisa functions) the elements of the cache can be read and written.

```python
buf.read(index)
buf.write(index, value)
```

In addition, buffers of type int/float support atomic operations that update the value in the buffer atomically and 
return its original value.

```python
buf.atomic_exchange(idx, desired) # update to desired
buf.atomic_compare_exchange(idx, expected, desired)
  # update to (old == expected ? desired : old)
buf.atomic_fetch_add(idx, val) # update to (old + val)
buf.atomic_fetch_sub(idx, val) # update to (old - val)
buf.atomic_fetch_and(idx, val) # update to (old & val)
buf.atomic_fetch_or(idx, val) # update to (old | val)
buf.atomic_fetch_xor(idx, val) # update to (old ^ val)
buf.atomic_fetch_min(idx, val) # update to min(old, val)
buf.atomic_fetch_max(idx, val) # update to max(old, val)
```

### Texture Type (`luisa.Texture2D`)

A texture is used to store a two-dimensional image on the device. The size of a texture is width x height x number of 
channels, where the number of channels can be 1, 2 or 4. Since storing by 3 channels significantly affects performance, 
if you only need to use 3 channels, create a 4-channel texture and use its first 3 components.

