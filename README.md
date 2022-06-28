# LuisaCompute

[![build](https://github.com/Mike-Leo-Smith/LuisaCompute/actions/workflows/ci.yml/badge.svg)](https://github.com/Mike-Leo-Smith/LuisaCompute/actions/workflows/ci.yml)

High-performance cross-platform computing framework for computer graphics and more.

## Roadmap

- ✅ Done and fully functional
- ⚠️ Done but with minor issues
- 🚧 Working in progress
- ⌛ Planned but not started
- ⏩ Not required/applicable

### Frontends/DSLs

| Implementation                 | Status                                    |
| ------------------------------ | ----------------------------------------- |
| C++                            | ✅                                         |
| Python                         | ⚠️ (no support for polymorphic constructs) |
| Custom Script/Shading Language | ⌛                                         |

### AST/IR

| Module   | Status                                                       |
| -------- | ------------------------------------------------------------ |
| AST      | ✅                                                            |
| IR       | 🚧 (inter-convertibility with the AST; maybe optimization passes) |
| AutoDiff | ⌛ (reverse mode; transformation passes on the IR)            |

### Runtime

| Module             | Status                                                       |
| ------------------ | ------------------------------------------------------------ |
| Device Interface   | 🚧 (re-designing bindless resource APIs; support for device-specific extensions) |
| Command            | 🚧 (re-designing bindless resource commands; serialization)   |
| Command Scheduling | ✅                                                            |
| Interoperability   | 🚧 (complete support for `native_handle` in progress; registration of external resources) |
| GUI Support        | 🚧 (re-designing swap-chain APIs)                             |

### Backends

| Implementation | Codegen Status                                               | Runtime Status                                               |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CUDA           | ✅                                                            | ✅                                                            |
| DirectX        | ✅ (simulated `atomic<float>` support)                        | ✅                                                            |
| Metal          | ✅ (simulated `atomic<float>` support)                        | ⚠️ (acceleration structure compaction disabled due to Metal bugs) |
| Vulkan         | 🚧 (will translate to SPIR-V)                                 | ⚠️ (needs testing)                                            |
| ISPC           | ⚠️ (no support for shared memory or `synchonize_block`; compiles very slow; simulated `atomic<float>` support) | ✅                                                            |
| LLVM           | ⚠️ (scalar only; no support for shared memory or `synchronize_block`; simulated `atomic<float>` support)<br />🚧 (vectorization, will bring support for shared memory and `synchronize_block` together) | ✅                                                            |
| Remote         | ⏩ (forwarded to underlying backends)                         | 🚧 (depends on serialization and networking)                  |

### Libraries/Applications

- 🚧 [LuisaRender](https://github.com/LuisaGroup/LuisaRender.git) (support for volumetric rendering, out-of-core tracing
  and shading, advanced sampling algorithms, custom shading language, Blender/C4D exporters, etc.)
- 🚧 [LuisaShaderToy](https://github.com/Mike-Leo-Smith/LuisaShaderToy.git) (custom shading language support planned)
- ⌛ Luisa Performance Primitives (pre-tuned kernels for high-performance sorting, mapping, reduction, image processing
  operators, etc.)

