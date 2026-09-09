# 逐操作归约策略：独立验证记录

2026-09-08，Apple M1 Max。此记录验证归约语义与编译边界，不报告性能加速。

## 源码和依赖边界

基于 `84afe85f3`，将准备提交的 index 导出到
`/tmp/luisa-reduction-checkpoint.gQUERp/source`，单独配置并完整构建
`/tmp/luisa-reduction-checkpoint.gQUERp/build`。配置为 RelWithDebInfo、SIMD/Metal/TIRx
启用、Metal4/unity build 关闭，LLVM 21。依赖子模块沿用本地 checkout（其中有未提交
修改），没有宣称整套依赖是干净源码状态。

TVM 固定在 `c7b458e946bc4266915da582457476bdcd9705ae`，FFI 固定在
`12dbf053b3d9ba4ebd9da3123b1aeca79cf74229`。本地 compiler/runtime 同步完整重建，
包含既有 MPP 扩展和新的 `metal-precise-math-v1.patch`；对应构建位于
`/Users/mike/.cache/luisa-tile/tvm-fragment-build.0ToXYr`。precise-math 扩展独立于 MPP，
只为 TVM 自身 Metal runtime 传递精确编译要求；Luisa Runtime 的 Metal 路径不需要该扩展。

## 执行结果

| 记录 | 结果 | 含义 |
|---|---|---|
| [独立完整构建](checkpoint-full-build.log) | exit 0 | 不是 target-only build |
| [独立 Tile CTest](checkpoint-all-tile.log) | 35/35，225.83 s | 此次提交的源码快照 |
| [Python benchmark 测试](checkpoint-python-tests.log) | 110/110 | 含元数据、完整输出和配对统计检查 |
| [未安装 precise 扩展的 TVM](checkpoint-unpatched-tvm.log) | 2/2 | Metal values/execution 验证严格策略明确拒绝、其他路径仍可执行 |
| [严格 Sphinx 构建](checkpoint-sphinx.log) / [本地链接检查](checkpoint-doc-links.log) | 通过 | API XML 已生成，未屏蔽缺失 XML 警告 |
| [原工作区 Tile CTest](all-tile-regression.log) | 34/35 | 另一个 matrix 实验仍有四项结构断言失败 |

独立验证命令：

```sh
cmake --build /tmp/luisa-reduction-checkpoint.gQUERp/build -j 6
ctest --test-dir /tmp/luisa-reduction-checkpoint.gQUERp/build --output-on-failure -j 2 -R '^test_tile'
uv run --offline --no-project --python 3.13 --with numpy --with torch \
  python -m unittest discover -s scripts/benchmark/tile_torch -p 'test_*.py'
```

原工作区的 matrix initializer/view 实验没有混入该提交，也没有通过删除测试或修改
预期值来消除失败。历史 device/threadgroup fence 问题已恢复正确的 publication 范围。

测试涵盖 DSL/IR policy、分析失效、默认无序树、显式保序树与左右 fold、空域、二维
遍历、非 identity 与正负零 seed、FP32 消去反例、混合严格/宽松归约、最终编译 fast math
和 Metal 模块序列化后再执行。归约 fixture 使用 FP32 bit-pattern 对照，不是放宽容差。
保序树目前仍由串行遍历实现；XIR 尚无一般 within-Tile tree/packet emitter。

兼容性复测使用相同 Luisa 二进制，仅将 `DYLD_LIBRARY_PATH` 指向
`/tmp/luisa-tvm-mpp.VaKmzx/build/lib` 的旧 TVM compiler/runtime：它包含既有 MPP 扩展，
但不包含 precise-math 扩展。不能把严格用例的预期编译拒绝算成 GPU fold 已执行。
[动态库加载记录](checkpoint-unpatched-loader.log) 确认五个 TVM 动态库都来自这个旧目录；
该次精确筛选运行了一个 fold 用例的 36 项断言，另十个注册用例按筛选要求跳过，
不能将其误报为十一项用例全部执行。上表的完整两个 CTest 没有这个筛选。

## 下一步

语义许可与硬件候选已经分开，但 composed-group 参考绑定不是校准过的全程序模型。
应继续测量 phase 的贡献工作、内存访问/转换、live state 和同步。尤其小 M contraction
被 8×8 atom 限制挡住时，要比较不同输出/归约方向分布，而不是依靠算子名字或降低精度。
