
# Roadmap

We list some possible future work here. If you have any ideas and suggestions, welcome to open an [issue](https://github.com/LuisaGroup/LuisaCompute/issues) or start a [discussion](https://github.com/LuisaGroup/LuisaCompute/discussions)!

 - ✅ Done and fully functional
 - ⚠️ Done but with minor issues
 - 🚧 Working in progress
 - ⌛ Planned but not started
 - ⏩ Not required/applicable

## Frontends

| Implementation                 | Status                                      |
|--------------------------------|---------------------------------------------|
| C++                            | ✅                                          |
| Python                         | ⚠️ (no support for polymorphic constructs)  |
| Custom Script/Shading Language | ⌛                                          |

## DSL

| Module   | Status                                                            |
|----------|-------------------------------------------------------------------|
| AST      | ✅                                                                  |
| XIR      | 🚧 (AST inter-convertibility and optimization passes)              |
| AutoDiff | 🚧 (forward/reverse mode transformation passes on XIR)             |

## Runtime

| Module             | Status                                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------------------------|
| Device Interface   | 🚧 (re-designing bindless resource APIs; support for device property query; support for device-specific extensions) |
| Command            | 🚧 (re-designing bindless resource commands; serialization)                                                         |
| Command Scheduling | ✅                                                                                                                   |
| Interoperability   | 🚧 (complete support for `native_handle` in progress; registration of external resources)                           |
| GUI Support        | 🚧 (re-designing swap-chain APIs)                                                                                   |
| AOT Support        | ⌛ (PSO caching and thin runtime)                                                                                    |
| Rasterization Support | 🚧 (Designing API) |

## Backends
| Implementation         | Codegen Status                                                                                                                                                                                     | Runtime Status                                                    |
 |------------------------|------------------------------------------|---------------------------------|
 | CUDA                   | ✅                                                                                                                                                                                                  | ✅                                                                 |
 | DirectX                | ✅ (simulated `atomic<float>` support)                                                                                                                                                              | ✅                                                                 |
 | Metal                  | ✅ (simulated `atomic<float>` support)                                                                                                                                                              | ⚠️ (acceleration structure compaction disabled due to Metal bugs) |
 | Vulkan                 | 🚧 (will translate to SPIR-V)                                                                                                                                                                      | ⚠️ (needs testing)                                                |
 | LLVM                   | ✅ (scalar only; simulated `atomic<float>`, shared_memory, and `synchronize_block` support)<br />🚧 vectorization | ✅                                                                 |
