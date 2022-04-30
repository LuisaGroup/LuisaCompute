# Python-Luisa 用户文档
多后端高性能计算库。

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

**注意：** 在 Python 终端交互模式（REPL）下不可使用（见[#10](https://github.com/LuisaGroup/LuisaCompute/issues/10)）。在 Jupyter notebook 中出错时可能导致崩溃（见[#11](https://github.com/LuisaGroup/LuisaCompute/issues/10)）。建议使用文件脚本，如`python3 a.py`。

```python
import luisa # 引入Luisa库，初始化
import numpy as np

b = luisa.Buffer(100, dtype=int)
# 声明函数 kernel
@luisa.kernel
def fill(x: int): # 括号内为参数列表
    b.write(dispatch_id().x, x)
    # b从外部捕获
    # dispatch_id 为内建函数，返回 int3 类型

fill(42, dispatch_size=(100,1,1))
# 并行执行，必须指定并行度 dispatch_size
res = np.ones(100, dtype='int32')
b.copy_to(res)
print(res)
```
## Kernel 与 Callable

kernel 是可由python(host)中并行调用的函数，并行度由 `dispatch_size` 参数指定。

callable 是可由kernel/callable调用的函数，其不可在python中直接调用。callable 中返回值类型必须统一，即不能出现多处return值类型不同的情形。

kernel/callable可以接受参数。参数列表可为空，由逗号隔开，每一项必须标记类型，形为`name: type`。其中，`name`为参数名字，`type` 为类型标记（见“类型”一节）。

## 类型

### 标量类型

- 32位有符号整形 `int`
- 单精度浮点数 `float`
- 逻辑类型 `bool`

### 向量、矩阵类型

`luisa.float3`, `luisa.float3x3` , ...

注：向量、矩阵尺寸只支持2~4，矩阵为方阵。向量元素可以为三种标量中的一种，矩阵只支持float矩阵。

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

```python3
# 声明了一个类型
# 其中dtype为标量/向量/矩阵类型，size为数组大小，通常较小（几千以内）。
arr_t = luisa.ArrayType(dtype, size)
# arr_t 可以作为kernel/callable参数列表中的类型标记
# 生成一个实例
a1 = arr_t() # 暂时只在kernel中支持
a2 = arr_t([value1, ...]) # 暂时只在python(host)中支持
# 访问成员
a1[idx] = value1 # python/kernel 都支持
```

### 结构体类型

```python
# 声明了一个类型
# 其中dtype为标量/向量/矩阵/数组/结构体类型
struct_t = luisa.StructType(name1=dtype1, name2=dtype2, ...)
# struct_t 可以作为kernel/callable参数列表中的类型标记
# 生成一个实例
a1 = struct_t() # 暂时只在kernel中支持
a2 = struct_t(name1=value1, ...) # 暂时只在python(host)中支持
# 访问成员
a1.name1 = value1 # python/kernel 都支持
```

（高级用法）结构体可以定义callable方法：

```python
@luisa.callable_method(struct_t)
def method1(self: luisa.ref(struct_t), ...):
    ...
```

如方法名为`__init__`，该结构体在kernel/callable中构造时会调用该方法。

### 引用

Callable的参数可以标记为引用类型`luisa.ref(type)`，例如：

```python
@luisa.callable
def flipsign(x: luisa.ref(int)):
    x = -x
```

参数类型可以为标量、向量、矩阵、数组、结构体的引用。注意Buffer等资源类型在传参过程中并不会复制数据，将资源类型作为参数时请勿标记为引用。

kernel不支持引用参数。

### Buffer类型

在设备上的数组，不能直接在python中访问其元素。暂时只支持以标量或向量为元素的Buffer。

Buffer和Array的区别是，Buffer是一种资源，由所有线程共享，长度可以很大；而Array是一个长度固定且较小的变量类型。

类型标记：`luisa.BufferType(dtype)`

创建buffer：`luisa.Buffer(size, dtype)`

kernel方法：

`read(idx)`

`write(idx, value)`

python方法：

`copy_to(arr)`

`copy_from(arr)`

元素为标量的buffer可以上传/下载到对应类型的numpy.array，注意必须使用int32/float32，而不是默认的64位类型。

### Texture2D类型

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

### Accel类型

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
hit.miss()
hit.inst # instance ID of the hit object
hit.prim # primitive ID of the hit object
```

TODO: `interpolate(hit, a,b,c)`

TODO: `offset_ray_origin(p,n)`

TODO: `offset_ray_origin(p,n,w)`

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

## 变量

在kernel/callable中初次出现（请勿与外部变量重名，TODO）的变量为局部变量。局部变量的类型在整个函数中保持不变。例如：

```python
@luisa.kernel
def fill():
    a = 1 # 定义了int类型的局部变量
    a = 2 # 赋值
    a = 1.5 # 禁止这么做，因为改变了类型
```

## 语法参考

...

# Python-Luisa 开发者文档

`src/py/lcapi.cpp` 导出PyBind接口到名为 lcapi 的库

`src/py/luisa` python库

`src/py/luisa/astbuilder` 遍历Python AST，调用FunctionBuilder

## LuisaCompute API

暂无文档。见`src/py/lcapi.cpp` 指向的c++函数

## AST变换

对AST的表达式节点维护两个属性：

`node.dtype` 表达式值的类型，见用户文档“类型”一节。调用 `luisa.types.to_lctype(node.dtype)` 可转换为 `lcapi.Types`

`node.expr` 表达式，类型为 `lcapi.Expression`

## 注

