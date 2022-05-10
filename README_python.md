# Python-Luisa 用户文档
Luisa 是一个嵌入Python的领域专用语言（DSL），面向高性能图形渲染编程，支持 Windows、Linux、MacOS 操作系统，支持多种计算后端，包括 CUDA、DirectX、Metal、ISPC。

**本项目尚在开发中，如遇问题或功能建议请[提交issue](https://github.com/LuisaGroup/LuisaCompute/issues)。**

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

您可以从下面这个简单的程序看到Luisa的大致用法。这个程序创建了10个线程，每个线程向一个缓存里的对应位置写入了42这个值。

```python
import luisa

luisa.init("cuda") # 如不提供参数，将自动选择后端
b = luisa.Buffer(10, dtype=int)

@luisa.func
def fill(x):
    b.write(dispatch_id().x, x)
    
fill(42, dispatch_size=10) # 并行执行fill函数
print(b.numpy()) # 输出：[42 42 42 42 42 42 42 42 42 42]
```
## Luisa函数

Luisa函数是用于进行大量运算的设施。

使用修饰符 `luisa.func` 可以将一个函数标记为Luisa函数。编写Luisa函数使用的语法与Python语法相同，尽管使用上稍有差别，见语法参考；一个Luisa函数在被调用时会被即时编译为静态类型的代码，从而在后端设备上运行。

一个Luisa函数可以被Python代码**并行地**调用，在Python代码上调用时，必须指定参数 `dispatch_size`，即并行线程的数量。例如，在上例中并行地调用了100个`fill`函数的线程，每个线程的代码相同，但`dispatch_id()`不同。Luisa函数也可以被另一个Luisa函数调用，此时即与Python中函数调用的方法相同。

> 可以这样理解：Python代码是宿主端的代码，在CPU上运行。Luisa函数是设备端的代码，在加速设备上运行（加速设备可以是GPU，也可以是CPU自身）。宿主端用于控制发出多个设备端的线程，以及初始化与收集数据；而设备端用于执行主要的运算。
>
> 设备端的运算默认是异步的，这意味着在设备端运算的过程中，宿主端可以继续进行其它操作而不必等待。宿主端调用 luisa.synchronize() 可以强制同步设备端，等待设备端完成运算。
>
> 如果您熟悉CUDA编程，luisa.func 对应于CUDA中的 \_\_device\_\_ 或 \_\_host\_\_ （取决于它在哪里被调用）

Luisa函数的参数可以有（但不要求）类型标记，如 `def f(a: int)`。如提供了类型标记，该函数在被调用时会检查传入参数的类型与对应标记一致。仅支持单一类型标记。

注意：在Python中并行调用一个Luisa函数时，作为参数传入的变量并不会被这个函数内部的代码修改。在Luisa函数中调用另一个Luisa函数时，函数的传参规则与Python一致，即：标量类型按值传递，其它类型的**对象按引用传递**。为了避免歧义，禁止在函数内给引用传递的参数直接赋值。

在Python中并行调用一个Luisa函数时，不支持函数返回值。函数输出的方式只能是向缓存或贴图（见下一节）写入数据。

目前Luisa函数只支持位置参数，不支持关键词参数，且不支持参数默认值。

## 类型

与Python不同，Luisa函数是**静态类型**的，即一个变量被定义后不能被重新赋值为其他类型的值。我们提供了如下几种类型：

标量、向量、矩阵、数组、结构体是纯数据类型，其底层表示为存储空间中固定大小的一块内存。可以在luisa函数中创建这些类型的局部变量。其构造与赋值均为**按值复制**，传参规则见上一节。通常来说，这些类型在设备端（Luisa函数）与宿主端（Python代码）有统一的操作方式。

缓存、贴图、资源索引、加速结构是资源类型，用于存储所有线程共享的资源。在luisa函数中可以引用资源，但不能创建资源类型的局部变量。这些类型在设备端（Luisa函数）与宿主端（Python代码）操作方式可能不同，通常来说，宿主端负责在运算开始前初始化资源中的数据，在运算结束后收集资源中的数据；而设备端会使用资源中的元素。

### 标量

- 32位有符号整形 `int`
- 单精度浮点数 `float`
- 逻辑类型 `bool`

注意 Luisa 函数中 int/float 精度相比 python 中的64位 int/float 精度较低。

### 向量

存储了固定个数的标量，其概念对应于数学上4D以内的列向量。

```
luisa.int2, luisa.bool2, luisa.float2,
luisa.int3, luisa.bool3, luisa.float3,
luisa.int4, luisa.bool4, luisa.float4
```

为了使用方便，您可以将向量和矩阵类型导入命名空间：

```python3
from luisa.mathtypes import *
```

n维的向量（n∈{2,3,4}）可以从1个或者n个对应类型的标量构造，例如

```python
float3(7) # [7.0, 7.0, 7.0]
bool2(True, False) # [True, False]
```

使用下标或者成员可以访问向量中的元素：

```python
a[0] == a.x
a[1] == a.y
a[2] == a.z # 仅3维及以上向量
a[3] == a.w # 仅4维向量
```

向量类型具有一类只读的swizzle成员，可以以任意顺序将其2~4个成员组成新的向量，例如

```python
int3(7,8,9).xzzy # [7,9,9,8]
```

### 矩阵

存储了固定个数的标量，其概念对应于数学上4D以内的方阵。

```
luisa.float2x2
luisa.float3x3
luisa.float4x4
```

注意，只支持浮点数类型的方阵。

为了使用方便，您可以将向量和矩阵类型导入命名空间：

```python3
from luisa.mathtypes import *
```

n×n的矩阵（n∈{2,3,4}）可以从1个对应类型的标量k构造，构造结果为k倍单位矩阵。也可以从n×n个对应类型的标量构造，顺序为列优先。例如：

```python
float2x2(4) # [ 4 0 ]
            # [ 0 4 ]
float2x2(1,2,3,4) # [ 1 3 ]
                  # [ 2 4 ]
                  # 打印结果为 float2x2([1,2],[3,4])
```

使用下标可以从矩阵中取出一个列向量，可读可写。例如：

```python
a = float2x2(4)
a[1] = float2(5) # a 变为 float2x2([1,2],[5,5])
```

### 数组 `luisa.Array`

数组可以存放固定数量、同一类型的若干元素，其元素类型可以是标量、向量或矩阵。

可以从一个列表构造数组，例如：

```python
arr = luisa.array([1, 2, 3, 4, 5])
```

由于数组是纯数据类型，可以在每个线程中作为局部变量创建，数组的大小通常较小（几千以内）。

使用下标可以访问数组中的元素，例如 `arr[4]`

### 结构体 `luisa.Struct`

结构体可以存放固定布局的若干成员（属性），每个成员的类型可以是标量、向量、矩阵、数组或另一个结构体。结构体的使用方式类似于Python中的类对象，但由于结构体是纯数据类型，不能动态增删成员。

使用若干个关键字参数构造一个结构体，例如：

```python
s = luisa.struct(a=42, b=3.14, c=luisa.array([1,2]))
```

使用属性运算符(.)可以访问一个结构体的成员，例如 `s.a`

注意，结构体的存储布局中的成员顺序与构造时给定参数的顺序一致，因此 `luisa.struct(a=42, b=3.14)` 不等同于 `luisa.struct(b=3.14, a=42)`

FIXME #45 

命名的结构体类型（见类型标记一节）可以将一个luisa函数作为方法（成员函数），例如：

```python
struct_t = luisa.StructType(...)
@luisa.func
def f(self, ...):
    ...
struct_t.add_method(f, "method_name") # 如果没有指定方法名，会自动取函数名作为方法名
```

这使得从该类型创建的结构体实例，在luisa函数中可以调用方法，形如 `a.method_name(...)`

可以定义结构体的构造函数：如方法名为`__init__`，luisa函数中创建该类型的结构体实例时会调用该方法。

### 缓存 `luisa.Buffer`

缓存是在设备上由所有线程共享的数组，其元素类型可以是标量、向量、矩阵、数组或结构体。不同于数组类型，缓存的长度可以任意大（受限于计算设备的存储空间）。

从列表或numpy array 创建一个缓存：

```python
luisa.buffer(arr)
```

该操作会在设备上创建一个相应元素类型和长度的缓存，并上传数据。如果`arr`是numpy array，其元素类型只能是`bool`、`np.int32`或 `np.float32`。

您也可以创建一个指定元素类型（见类型标记一节）和长度的缓存：

```python
luisa.Buffer.empty(size, dtype)  # 创建一个未初始化的缓存
luisa.Buffer.zeros(size, dtype)  # 创建一个缓存，其中每个元素均初始化为 dtype(0)
luisa.Buffer.ones(size, dtype)   # 创建一个缓存，其中每个元素均初始化为 dtype(1)
luisa.Buffer.filled(size, value) # 创建一个缓存，其中每个元素均初始化为 value
```

可以从列表或numpy array 向已有的缓存上传数据：

```python
b.copy_from(arr) # arr 的数据类型和长度必须和 b 一致
```

不能直接在宿主端（Python代码中）直接访问缓存的元素。下载 TODO 

在设备端（Luisa函数中）可以对缓存的元素进行读写：

```python
b.read(index)
b.write(index, value)
```

除此之外，缓存还支持原子操作 TODO

### 贴图 `luisa.Texture2D`

贴图用于在设备上存储二维的图像。一个贴图的尺寸为宽度×高度×通道数，其中通道数可以是1、2或4。由于按3通道存储会影响性能，如果您只需使用3个通道，请创建4通道的贴图并使用其前3个分量。

从 numpy array 创建一个贴图：

```python
luisa.texture2d(arr) # arr 形状为 (width, height, channel)
```

该操作会在设备上创建一个相应元素类型和尺寸的贴图，并上传数据。`arr`的元素类型只能是`np.int32`或 `np.float32`。

您也可以创建一个指定元素类型（见类型标记一节）和尺寸的贴图：

```python
luisa.Texture2D.empty(w, h, channel, dtype, [storage])  # 创建一个未初始化的贴图
luisa.Texture2D.zeros(w, h, channel, dtype, [storage])  # 创建一个贴图并初始化为0
luisa.Texture2D.ones(w, h, channel, dtype, [storage])   # 创建一个贴图并初始化为1
luisa.Texture2D.filled(w, h, value, [storage])
# 如果value是标量，则创建1通道的贴图；如果value是2/4维向量，则创建2/4通道的贴图；并使用value初始化
```

其中storage是可选参数 TODO 见 luisa.PixelStorage。如未指定，默认使用最高精度（和dtype一样的精度）。

不能直接在宿主端（Python代码中）直接访问贴图的元素。可以将贴图下载到对应形状和类型的numpy array，或直接调用`numpy`方法返回下载结果。

```python
tex.copy_to(arr) # arr 形状为 (width, height, channel), 类型 np.int32 / np.float32
arr1 = tex.numpy()
```

在设备端（Luisa函数中）可以对贴图的元素进行读写：

```python
b.read(index)
b.write(index, value)
```

其中index为int2类型，表示贴图上的坐标(x,y)。对于2/4通道的贴图，读写值以向量为单位。

### 资源索引 `luisa.BindlessArray`

资源索引是设备上的一张表，其每个元素存储了资源描述符，可以用于索引缓存和浮点类型的贴图。索引的默认大小`n_slots`为65536。可以从一个字典创建资源索引，字典的每一项中，键为0~n_slots-1的整数，值为一个缓存或浮点类型的贴图，例如：

```python3
luisa.bindless_array({0: buf1, 42: tex1, 128: buf2})
```

也可以创建一个空的资源索引；可以向资源索引中插入索引项或删除索引项，但注意改变索引后，在使用前必须调用`update`:

```python
a = luisa.BindlessArray.empty(131072) # 参数 n_slots 可选，默认 65536
a = luisa.BindlessArray(131072) # 同上，别名
a.emplace(index, res) # 在index位置放置res资源的索引
a.remove_buffer(index) # 删除index位置的缓存索引
a.remove_texture2d(index) # 删除index位置的贴图索引
a.update() # 更新索引表
res in a # 查询该资源是否在表中
```

在设备端（Luisa函数中）可以读取被索引的资源：

```python
a.buffer_read(element_type, idx, element_idx) # 返回 element_type
a.texture2d_read(idx, coord) # 返回float4
a.texture2d_sample(idx, uv) # 返回float4
a.texture2d_size(idx) # 返回float4
```

注：目前的用法可能有点奇怪？可能后续会更改

### 加速结构 `luisa.Accel`

加速结构是在设备上的用于加速计算射线与3D空间中物体交点的数据结构。

TODO

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
hit.interpolate(a,b,c)
```

TODO: `interpolate(hit, a,b,c)`

TODO: `offset_ray_origin(p,n)`

TODO: `offset_ray_origin(p,n,w)`

## 类型标记

TODO

## 类型规则

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

