# Python-Luisa 用户文档
多后端高性能计算库，支持 ISPC/CUDA/DirectX/Metal 后端。

## 编译与运行

```bash
cmake -S . -B build_debug -G Ninja -D CMAKE_BUILD_TYPE=Debug -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -D LUISA_COMPUTE_ENABLE_PYTHON=ON -D LUISA_COMPUTE_ENABLE_LLVM=OFF -D LUISA_COMPUTE_ENABLE_ISPC=ON -D LUISA_COMPUTE_ENABLE_METAL=OFF
cmake --build build_debug
```
此编译命令会在`build_debug`下生成可被python引入的luisa库。

运行测试脚本：

```bash
cd build_debug/bin
python3 test.py
```

## 语言用例

```python
import luisa # 引入Luisa库，初始化
import numpy as np

b = luisa.Buffer(100, dtype=int)

@luisa.func
def fill(x):
    b.write(dispatch_id().x, x)
    # dispatch_id 为内建函数，返回 int3 类型

fill(42, dispatch_size=(100,1,1))
# 并行执行，指定并行线程数量为 dispatch_size

res = np.ones(100, dtype='int32')
b.copy_to(res)
print(res)
```
## Luisa函数

使用修饰符 `luisa.func` 可以将一个函数标记为 Luisa 函数，这使得该函数可以在后端设备上运行。

一个 Luisa 函数可以被host端并行调用，也可以被另一个Luisa函数调用。在被调用时，该函数会被即时编译为后端设备可以执行的代码。

函数的参数可以有（但不要求）类型标记，如 `def f(a: int)`。如有类型标记，在调用时会检查对应参数类型。

在Luisa函数中调用函数时，函数的传参规则与Python一致，即：标量类型按值传递，其它任何类型按引用传递（但不可以被赋值）。在python中并行调用时，luisa函数无论如何都不会修改python端传入的参数（这是显然的）。

注：Luisa函数只支持位置参数，不支持关键词参数，且不支持参数默认值。

## 类型

### 标量类型

- 32位有符号整形 `int`
- 单精度浮点数 `float`
- 逻辑类型 `bool`

注意 Luisa 函数中 int/float 精度相比 python 中的64位 int/float 精度较低。

### 向量、矩阵类型

`luisa.float3`, `luisa.float3x3` , ...

向量、矩阵尺寸只支持2~4，矩阵为方阵。向量元素可以为三种标量中的一种，而矩阵元素只能是float。

导入命名空间：

```python3
from luisa.mathtypes import *
```

向量类型具有一类swizzle成员，可以以任意顺序将其成员组成新的向量，例如

```python
v1 = int3(7,8,9)
v1.x # 7，可读可写
v1.xzzy # int4(7,9,9,8)，只读
```

矩阵类型暂不支持访问元素。

### 数组类型

===待更新===

数组可以存放固定数量、同一类型的若干元素。

```python3
# 声明了一个类型
# 其中dtype为标量/向量/矩阵类型，size为数组大小，通常较小（几千以内）。
arr_t = luisa.ArrayType(dtype, size)
# arr_t 可以作为func参数列表中的类型标记
# 生成一个实例
a1 = arr_t() # 暂时只在kernel中支持
a2 = arr_t([value1, ...]) # 暂时只在python(host)中支持
# 访问成员
a1[idx] = value1 # python/kernel 都支持
```

注意：数组的构造和赋值均为按值复制

### 结构体类型

===待更新===

```python
# 声明了一个类型
# 其中dtype为标量/向量/矩阵/数组/结构体类型
struct_t = luisa.StructType(name1=dtype1, name2=dtype2, ...)
# struct_t 可以作为func参数列表中的类型标记
# 生成一个实例
a1 = struct_t() # 暂时只在kernel中支持
a2 = struct_t(name1=value1, ...) # 暂时只在python(host)中支持
# 访问成员
a1.name1 = value1 # python/kernel 都支持
```

注意：结构体的构造和赋值均为按值复制

结构体可以将一个luisa函数作为方法：

```python
@luisa.callable
def f1(self, ...):
    ...
struct_t.add_method(f1)
```

如方法名为`__init__`，该结构体在luisa函数中构造时会调用该方法。

### Buffer类型

===待更新===

在设备上的数组，不能直接在python中访问其元素。其元素类型可以是标量、向量、矩阵、数组或结构体。

Buffer和Array的区别是，Buffer是一种资源，由所有线程共享，长度可以很大；而Array是一个长度固定且较小的变量类型，可以作为每个线程局部变量的类型。

类型标记：`luisa.BufferType(dtype)`

创建buffer：`luisa.Buffer(size, dtype)`

kernel方法：

`read(idx)`

`write(idx, value)`

python方法：

`copy_to(arr)`

`copy_from(arr)`

元素为标量的buffer可以上传/下载到对应类型的numpy.array，注意必须使用int32/float32，而不是默认的64位类型。

当元素类型为向量时，需要使用对应长度的标量类型的numpy.array。注意：由于对齐要求，长度为3的向量占用4个对应类型标量的空间，矩阵float3x3占用12个float32的空间。例如

```python
b = luisa.Buffer(100, luisa.float3)
arr = numpy.zeros(400, dtype=numpy.float32)
b.copy_from(arr)
```

TODO: 需要提供用户友好的上传下载方式，以及支持其它类型元素的buffer

`copy_from` 可以从长度和元素类型一致的列表（list）上传到buffer。

### Texture2D类型

===待更新===

在设备上的二维贴图。不能直接在python中访问其元素。

创建贴图：`luisa.Texture2D(width, height, channel, dtype, [storage])`

channel: 1/2/4

dtype: int/float

storage 可选，像素存储格式，见 luisa.PixelStorage。如未指定，默认使用dtype同样精度。

作为参数的类型标记：`luisa.Texture2DType(dtype)`

kernel方法：

`read(idx)`

`write(idx, value)`

python方法：

`copy_to(arr)`

`copy_from(arr)`

上传/下载到storage对应类型的numpy.array。

### BindlessArray类型

===待更新===

可以放置多个 buffer / texture2d(float) 的容器。

创建：

```python
a = luisa.BindlessArray()
a.emplace(123, luisa.Buffer(...))
a.emplace(456, luisa.Texture2D(..., dtype=float))
a.remove_buffer(123)
a.remove_texture2d(456)
res in a
a.update()
```

进入luisa函数使用前需要调用update

作为参数的类型标记：`luisa.BindlessArray`

luisa函数内方法：

```python
a.buffer_read(element_type, idx, element_idx)
a.texture2d_read(idx, coord)
a.texture2d_sample(idx, uv)
a.texture2d_size(idx)
```

注：目前的用法有点奇怪，可能后续会更改

### Accel类型

===待更新===

在设备上的（光线求交）加速结构。

构建例子：

```python
accel = luisa.Accel()
v_buffer = luisa.Buffer(3, float3)
t_buffer = luisa.Buffer(3, int)
v_buffer.copy_from(np.array([0,0,0,0,0,1,2,0,0,2,1,0], dtype=np.float32))
t_buffer.copy_from(np.array([0,1,2], dtype=np.int32))
mesh = luisa.Mesh(v_buffer, t_buffer)
accel.add(mesh)
accel.build() 
```

作为参数的类型标记：`luisa.Accel`

kernel/callable 方法：

TODO: `make_ray(origin, direction)`

TODO: `make_ray(origin, direction, t_min, t_max)`

```python
ray.t_min
ray.t_max
ray.get_origin()
ray.get_direction()
ray.set_origin(k)
ray.set_direction(k)
hit = accel.trace_closest(ray)
hit1 = accel.trace_any(ray)
hit.miss()
hit.inst # instance ID of the hit object
hit.prim # primitive ID of the hit object
```

TODO: `interpolate(hit, a,b,c)`

TODO: `offset_ray_origin(p,n)`

TODO: `offset_ray_origin(p,n,w)`

### 类型转换

类型转换规则？目前只支持标量、向量、矩阵的类型转换。

```python
int(a)
float3(b)
```

TODO: struct / array

### 运算的类型规则

二元运算符和二元内建函数在不同运算时的转换规则如下：

#### 标量运算

不同类型变量运算时会出现隐式类型转换。

> 如 `int + float -> float`。

#### 同维度的向量 / 矩阵运算

将运算符广播到向量 / 矩阵的各元素上，不支持不同类型变量运算。

> `float3 + float3` 表示两个 `float3` 的对应元素分别相加。
> 
> `int3 + float3` 非法。

#### 标量与向量的运算

将运算符广播到向量的各元素上，不支持不同类型变量运算。

> `float + float4` 表示 `float` 标量与 `float4` 的每一个元素分别相加。 
>
> `int + float4` 非法。

## 内建函数与方法

kernel/callable中可以调用内置的函数。

```python
# 线程信息
dispatch_id(), dispatch_size() # 返回int3
# 打印
print(value1, "wow", ...) # 暂时不支持 formatted string
# 数学函数
isinf(x), isnan(x)
acos(x), acosh(x), asin(x), asinh(x), atan(x), atanh(x)
cos(x), cosh(x), sin(x), sinh(x), tan(x), tanh(x)
exp(x), exp2(x), exp10(x), log(x), log2(x), log10(x)
sqrt(x), rsqrt(x)
ceil(x), floor(x), fract(x), trunc(x), round(x)
# 向量矩阵创建
make_bool3(x,y,z), ...
make_float2x2(...)
```

一些类型具有可调用的方法，见类型对应文档。

### 数学函数

内建数学函数支持标量运算和向量运算。传入向量参数时会将函数广播至向量的各个维度上。

### 创建向量 / 矩阵的内置函数

对于标量类型 `T = int|uint|float|bool` 和长度 `N = 2|3|4`，都有 内置函数 `make_<T><N>`，表示创建一个 类型为 T、长度为 N 的向量。

> 如 `make_int2` 和 `make_float3`

`make_<T><N>` 支持灵活的参数类型，只需要保证所有参数的长度和等于 `N` 即可。 

> 如 `make_int3` 可以接受三个 `int`、一个 `int2` 和一个 `int`，或者一个 `int` 和一个 `int2`

如果传入的参数类型不为 `T`，那么它将被遵循 C 的隐式类型转换规则并转换为 `T`。

> 如调用 `make_int3(0.5, 1.5, 2.5)` 会得到 `int3(0, 1, 2)`

创建矩阵的内置函数与向量类似，区别在于它仅支持 `float` 类型，即只有 `make_float2x2, make_float3x3, make_float4x4` 三个。

`make_float<N>x<N>` 要求参数为以下三种之一：

- 一个 `float<N>x<N>` 类型
- `N` 个 `float<N>` 类型
- `N * N` 个 `float` 类型

`make_float<N>x<N>` 尚不支持隐式类型转换。

## 变量

局部变量的类型在整个函数中保持不变。例如：

```python
@luisa.kernel
def fill():
    a = 1 # 定义了int类型的局部变量
    a = 2 # 赋值
    a = 1.5 # 禁止这么做，因为改变了类型
```

同样地，如果函数的参数是一个类型，那么不能给这个参数赋另一个类型的值。（除了按值传的基本类型参数外，其它类型的参数不能被赋值）

## 语法参考

kernel中尚不支持 list, tuple, dict 等python提供的数据结构

### for 循环

仅支持 range for，形如 `for x in range(...)`

# Python-Luisa 开发者文档

## 概述

一个luisa.func在python中并行调用时，编译为kernel；在luisa.func中调用时，编译为callable。

## 文件结构

`src/py/lcapi.cpp` 导出PyBind接口到名为 lcapi 的库

`src/py/luisa` python库

`src/py/luisa/astbuilder` 遍历Python AST，调用FunctionBuilder

## LuisaCompute API

暂无文档。见`src/py/lcapi.cpp` 指向的c++函数

## 抽象语法树

使用Python提供的语法解析工具，可以将用户函数解析为一个Python抽象语法树（AST）。astbuilder模块递归地遍历这一语法树，在遍历过程中调用lcapi，以将该语法树转换为LuisaCompute的函数表示。

对于每一个kernel/callable，在遍历过程中，维护一些全局信息：

local_variable[变量名] -> (dtype, expr)

return_type

uses_printer

在递归遍历语法树过程中，对AST的每个表达式节点计算出两个属性：

`node.dtype` 该节点的类型标记，表示其表达式值的类型，见用户文档“类型”一节。

如果该节点的类型是一个数据类型（即标记类型为用户文档中除ref外的类型，而不是下述的“非数据类型标记”），那么调用 `luisa.types.to_lctype(node.dtype)` 可将类型标记转换为 `lcapi.Types`

`node.expr` 该节点的表达式。如果该节点的类型是一个数据类型，那么其表达式的类型为 `lcapi.Expression`；否则见下

`node.lr` 该节点是左值还是右值。`'l'` 或`'r'`

## 非数据类型标记

除用户文档中的类型标记外，AST节点的类型标记 `node.dtype` 还可以为以下值。这些值不可以作为 kernel/callable 的参数类型标记。

`type` 该节点表示的是一个类型，此时`node.expr`为对应的类型标记

`CallableType` 该节点表示的是一个callable，此时`node.expr`为callable

`BuiltinFuncType` 该节点表示的是一个内建函数，此时`node.expr`为一个字符串，内建函数的名字

`BuiltinFuncBuilder` 该节点表示的是一个内建函数，此时`node.expr`为一个函数` (argnodes)->(dtype,expr)`

`str` 该节点表示的是一个字符串，此时`node.expr`为一个字符串字面值。这种情况只允许在 `print` 函数的参数里出现

`list` 是一个列表；只在array的参数中出现。



## 注1

### 对传入参数的赋值语义

使赋值语义与python接近？
```python
@luisa.callable
def f(x): # x为int/float时按值传，x为vector/matrix/array/struct时按引用传
    # 例：x类型为float3
    x.y = 2 # 会改变调用者传入参数
    x[1] = 2 # 会改变调用者传入参数
    x += float3(1) # 会改变调用者传入参数
    x = x + float3(1) # 不会改变调用者传入参数：右侧x为传入参数，左侧x为新建的局部变量。
    # 这种情况警告用户
    # 不可以在非最外层scope中创建新变量覆盖旧变量！
    # 注：x = 1 也是可以的，允许新建不同类型的局部变量。同样警告用户
    x = float3(1) # 赋值。x仍为上一条语句创建的局部变量
    # 这句是普通赋值，赋值不可以改变类型，否则直接报错
```
如果在kernel中出现会改变传入参数（vector/matrix/array/struct）的语义，警告用户，实际上不会改变host中的值。

方案2：

```python
@luisa.callable
def f(x): # x为int/float时按值传，x为vector/matrix/array/struct时按引用传
    # 例：x类型为float3
    x.y = 2 # 会改变调用者传入参数
    x[1] = 2 # 会改变调用者传入参数
    x += float3(1) # 会改变调用者传入参数
    x = x + float3(1) # 禁止！为了防止歧义，用户不能给引用参数赋值
```

### callable对传入参数的修改/赋值语义

```python
def test_modify(a: int, b: Buffer..., c: float3):
    a += 1 # GOOD? (scalars are passed by value in python)
    b.write(...) # Good
    c.x = 4 # allow modify (reference)?
    c += 1 # allow modify (reference)?
    # view vector/struct as object?
def test_assign(a: int, b: Buffer..., c: float3):
    a = 1 # GOOD? (assignment. scalars are passed by value in python)
    b = ... # ???
    c = ... # ???
```

### kernel对传入参数的修改/赋值语义

```python
def test_modify(a: int, b: Buffer..., c: float3):
    a += 1 # GOOD? (scalars are passed by value in python)
    b.write(...) # Good
    c.x = 4 # BAD/WARN?
    c += 1 # BAD/WARN?
def test_assign(a: int, b: Buffer..., c: float3):
    a = 1 # GOOD? (assignment. scalars are passed by value in python)
    b = ... # ???
    c = ... # ???
```

### kernel/callable对捕获变量的修改/赋值语义

```python
a = 3
b = buffer(...)
c = float3()
def test_modify():
    a += 1 # BAD (automatically disables capture)
    b.write(...) # GOOD
    c.x = 4 # BAD
    c += 1 # BAD/WARN?  (automatically disables capture)
def test_assign():
    a = 1 # ???
    b = ... # ???
    c = ... # ???
```

注意：允许赋值覆盖原变量是危险的，可能出现一种情况：

```python
a = 3
def h():
    if cond:
        a = 4
    b = a
```

补注：如果一个名字被赋值了，那么就不会出现在closure_var中。

### 方案B

为每个变量记录创建的scope，禁止在更外层的scope使用？

