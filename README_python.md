# Python-Luisa 用户文档
### 编译与运行

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

### 语言用例

**注意：**在 Python 终端交互模式（REPL）下不可使用（见[#10](https://github.com/LuisaGroup/LuisaCompute/issues/10)）。在 Jupyter notebook 中出错时可能导致崩溃（见[#11](https://github.com/LuisaGroup/LuisaCompute/issues/10)）。建议使用文件脚本，如`python3 a.py`。

```python
import luisa
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
### 参数列表

参数列表可为空，由逗号隔开，每一项为`name: type`

`name`参数名字，`type` 为类型标记，见下

### 类型

##### 标量类型

`int`, `float`, `bool`

分别为32位有符号整形、单精度浮点数、布尔值。

##### 向量、矩阵类型

`luisa.float3`, `luisa.float3x3` , ...

注：向量、矩阵尺寸只支持2~4，矩阵为方阵。向量元素可以为三种标量中的一种，矩阵只支持float矩阵。

导入命名空间：

```python3
from luisa.mathtypes import *
```

##### 数组类型

```python3
arr_t = luisa.ArrayType(dtype, size)
```

其中dtype为标量/向量/矩阵类型，size为数组大小，通常较小（几千以内）。

##### 结构体类型

```python
struct_t = luisa.StructType(name1=dtype1, name2=dtype2, ...)
```

其中dtype为标量/向量/矩阵/数组/结构体类型

##### Buffer类型

暂时只支持标量Buffer

方法：

`read(idx)`

`write(idx, value)`

TODO

### 函数调用

##### 自定义函数

callable 是自定义的可被kernel调用的函数，其不可在python中直接调用。

callable 中返回值类型必须统一，即不能出现多处return值类型不同的情形。

##### 内建函数

##### 内建方法

### 变量

在kernel/callable中初次出现（请勿与外部变量重名，TODO）的变量为局部变量。局部变量的类型在整个函数中保持不变。例如：

```python
@luisa.kernel
def fill():
    a = 1 # 定义了int类型的局部变量
    a = 2 # 赋值
    a = 1.5 # 禁止这么做，因为改变了类型
```

### 调试

TODO



# Python-Luisa 开发者文档

`src/py/lcapi.cpp` 导出PyBind接口

`src/py/luisa` python库

