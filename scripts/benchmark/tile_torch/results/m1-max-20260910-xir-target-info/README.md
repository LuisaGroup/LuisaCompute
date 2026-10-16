# XIR backend target info / Metal4：接入与验证 checkpoint

日期：2026-09-10。Apple M1 Max；macOS 26 / Darwin 25.6；LLVM 22.1.8。
下方九个 Metal4 探针和 `test-final.xml` 对应实现提交 `214223013`：
源码基线 `7670fbd5769910a0c8c19ab5f60cdf9bc04f35a4` 加该实现提交的改动。
主分支为 `next`。隔离源码不包含用户原有 TIRx/matrix/iOS 等 WIP。

## 结论与边界

- `ExecutionTargetInfo` 提供实际 packet width、候选 block、附加合法性、调度模型和默认 cost policy。
  公共 solver 不再限制局部分布为 16 lanes；CPU home-chunk 模型保留在具体 thread-pool info 中。
  映射仍须满足当前 XIR 约束、power-of-two tree 和完整物理 packet ABI，cost 不能豁免语义检查。
- Metal4 接通 `TileIR -> XIR -> LLVM/AIR -> MetalShader -> Runtime`，没有 AST 回转或 MSL/Python 导出。
  真正的 32-lane 映射与程序/内存边界检查通过。不是 MPP/tensor-MMA 路径。
- GPU prior 没有校准；snapshot 容量尚未完整进入逐候选筛选，深层 unsupported/budget 拒绝仍有 fatal 边界。
  本次**没有**达到或证明 Torch/MPS parity，也没有纯 GPU kernel 时间。

## 验证

完整配置构建成功后，选定的 5 个 CTest 全部通过，见 `test-final.xml`：

| 测试 | 检查范围 | 断言数 |
|---|---|---:|
| test_tile_ir | 含最新 allocator / analysis holder 生命周期回归 | 340 |
| test_tile_xir_target_info | 32/64 lanes、65/129 尾部、成本/合法性隔离、CPU 兼容、无伪 FP64 literal | 642 |
| test_tile_xir_metal | 41 个 FP32 正例、5 个预期拒绝，所有输出/输入和哨兵 | 98,682 |
| test_tile_xir_runtime | SIMD 数值、视图、alias、根遍历等 | 5,085,845 |
| test_tile_xir_llm | SIMD normalization/activation/softmax/attention | 2,370,855 |

Metal4 正例包含 RMSNorm、LayerNorm、SwiGLU、GELU residual、masked softmax、RoPE，
宽度含 6/7/32/65/66/128/129/1024/1025，主要 rows=17；1-lane 与可实现的 32-lane 都检查。
宽度小于 32 的 forced local mapping 返回错误；automatic 回到完整程序映射。
不把 mock W64 的 IR 验证说成在本机运行了 64-lane 硬件。

配置开启 SIMD、Metal、Metal4、TIRx、DSL 和全部测试，GUI/fallback/CUDA 关闭，SYSTEM_STL。
`LUISA_COMPUTE_TILE_XIR_TEST_TIRX_COMPARISON=OFF`：本机 TVM 链接 LLVM21，而原生后端链接 LLVM22；
同进程比较在 LLVM analysis manager 崩溃。移除两个测试的 TVM 链接后，上述完整 SIMD oracle 通过；
独立 TIRx targets 仍构建。本次没有运行其全套测试，也没有声称修好两个 LLVM 主版本的共存。
未加载 TVM 的原生 Tile 诊断 `simd-no-tvm.json` 也通过；它在 literal 修正前采集，
仅用于环境排查，不用于性能结论，也不受下方最终测量二进制哈希覆盖。

8 个改动 C++ 编译单元通过项目 clangd/clang-tidy 检查（0 errors，保留已报告的风格建议）。
最终隔离工作区的 Doxygen + 严格 Sphinx 构建通过；72 个 HTML、5,420 个本地链接/资源、199 个兼容锚点检查通过。

验证中发现并处理：

1. 同步 overlay 保留旧 mtime，导致一次增量构建复用了旧 16-lane 对象；LLDB 证实机器码仍有 `cmp #0xf`。
   已对隔离源中全部 overlay 显式刷新时间并完整重建；本表只采用之后的结果。
2. UT 后端名白名单漏掉 `metal4`，首轮出现 0 asserts / 全部 skipped 的假通过。
   已修白名单，并让专用 CTest 对零断言/跳过直接失败；首轮不计通过。
3. FP32 literal 的 double 属性容器被错误降低为 FP64 cast，Metal 因而拒绝；现在直接生成声明精度的常量。
4. 上游 CUDA artifact 测试漏引入 `ceil_div` 声明，补齐 mathematics header。

旧 `test_tile_xir` 的部分非法 lower 负例仍与上游 fatal 错误策略不匹配；本次不将它计为通过。

### 合并最新 next 的复验

实现提交后合入 `f6334b385`，合并提交为 `5045809d9`。上游将测试 helper 改为
无异常 fatal 检查；合并保留该改动及 Metal4 benchmark 路径标识，没有更改 planner/lowering。
完整构建通过，但第一次复验为 4/5：SIMD LLM 中两个旧 `expect(throws(...))`
遇到新的 attention shape assertion 而中止，见 `test-next-merge-before-fix.xml`。

这两个负例改为独立 CTest 子进程，在创建设备/线程池前执行同样的非法输入。
包装器要求非零退出及 `Invalid attention shape` 诊断，明确拒绝超时和正常返回；
用 `/usr/bin/true` 的反向控制也确认正常退出不会被误计为通过。原有数值检查未删除。
此适配没有改变下面九个历史探针的数据，不应把历史二进制哈希解释为合并后产物。

最终源码为 `5045809d9` 加本节所在提交的负例适配，完整配置构建通过，7 项 CTest
全部通过（197.02 s），见 `test-next-rejection.xml`。Metal4 仍有 98,682 个断言，
SIMD Runtime 为 5,085,845；SIMD LLM 为 2,370,853，恰好少掉移至独立进程的两个
异常断言，其余数值检查不变。新增修改的 C++ 单元也通过 clangd/clang-tidy 和格式检查。

## 运行通路探针：不是可靠的性能排名

相同 FP32 128×1024 fixture，每次完整 FP64 oracle 检查两遍，输出两端各 17 个哨兵。
3 个样本、每批目标 20 ms、warmup 20 ms；顺序运行 1、32、0，没有 ABBA 或跨进程重复。
以下是 **synchronized Runtime host-wall batch throughput，µs/dispatch 中位数**，不是纯 kernel：

| 算子 | forced 1 | forced 32 | auto 0（都选 32） |
|---|---:|---:|---:|
| RMSNorm | 1281.50 | 215.84 | 555.85 |
| masked softmax | 2701.79 | 2488.89 | 6992.92 |
| SwiGLU | 5214.94 | 16261.71 | 275.71 |

相同物理 plan 的 forced32/auto 出现巨大差异，特别是 SwiGLU 约 59×。
因此这些数据只证明通路和正确性，并暴露计时/Runtime 排查需求；**不能据此宣称 32 lanes 更快、
自动 planner 更优或拟合 cost coefficients**。下一步需要 Metal4 自己的 GPU timestamp/feedback、
稳定的重复与对照，分离 kernel 和 dispatch；legacy Metal timing helper 不适用于 Metal4。

复现每个 JSON（输出路径必须尚不存在）：

```sh
LUISA_TILE_BENCH_XIR_BACKEND=metal4 LUISA_TILE_BENCH_XIR_LOCAL_LANES=32 \
  benchmark_tile_xir llm rmsnorm 128,1024 1 1 3 20 20 output.f32
```

文件名后缀是请求的 local_lanes；JSON realization 是实际选择。没有修改源 kernel。
原始导出输入/输出留在 `/tmp/luisa-xir-target-smoke.hkNajR`，未把二进制数组加入 Git。

本次测量二进制 SHA-256：

```text
benchmark_tile_xir              34176f5be19674c80da0bc8d41ed0b84ebc74245229944424df8ba1d86f89790
libluisa-backend-metal4.so      2b950e2e373a52cdf1180d26698f5ed9ce202e533df9920078753ac8f5fc7173
libluisa-tile-bridge-xir.dylib   fae5795fedf45ffadf204948f87a9afaf8238164dee6db0431b6dd4f9f749a35
libluisa-backend-simd.so        9c9028d51be01318b8e5ddaf1393795db67f407ce6f077f861718741203b4661
```
