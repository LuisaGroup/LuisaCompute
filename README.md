# LuisaCompute

High-performance cross-platform computing framework for graphics and beyond.

## Building and Running

See [BUILD](BUILD.md).

## Python Frontend

See [README-Python](README_Python.md).

## Roadmap

- ✅ Done and fully functional
- ⚠️ Done but with minor issues
- 🚧 Working in progress
- ⌛ Planned but not started
- ⏩ Not required/applicable

### Frontends/DSLs

| Implementation                 | Status                                     |
|--------------------------------|--------------------------------------------|
| C++                            | ✅                                          |
| Python                         | ⚠️ (no support for polymorphic constructs) |
| C API                          | ⌛ (for easy bindings in other languages)   |
| Custom Script/Shading Language | ⌛                                          |

### AST/IR

| Module   | Status                                                            |
|----------|-------------------------------------------------------------------|
| AST      | ✅                                                                 |
| IR       | 🚧 (inter-convertibility with the AST; maybe optimization passes) |
| AutoDiff | ⌛ (reverse mode; transformation passes on the IR)                 |

### Runtime

| Module             | Status                                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------------------------|
| Device Interface   | 🚧 (re-designing bindless resource APIs; support for device property query; support for device-specific extensions) |
| Command            | 🚧 (re-designing bindless resource commands; serialization)                                                         |
| Command Scheduling | ✅                                                                                                                   |
| Interoperability   | 🚧 (complete support for `native_handle` in progress; registration of external resources)                           |
| GUI Support        | 🚧 (re-designing swap-chain APIs)                                                                                   |
| AOT Support        | ⌛ (PSO caching and thin runtime)                                                                                    |

### Backends

| Implementation         | Codegen Status                                                                                                                                                                                     | Runtime Status                                                    |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| CUDA                   | ✅                                                                                                                                                                                                  | ✅                                                                 |
| DirectX                | ✅ (simulated `atomic<float>` support)                                                                                                                                                              | ✅                                                                 |
| Metal                  | ✅ (simulated `atomic<float>` support)                                                                                                                                                              | ⚠️ (acceleration structure compaction disabled due to Metal bugs) |
| Vulkan                 | 🚧 (will translate to SPIR-V)                                                                                                                                                                      | ⚠️ (needs testing)                                                |
| ISPC<br />[Deprecated] | ⚠️ (no support for shared memory or `synchonize_block`; compiles very slow; simulated `atomic<float>` support)                                                                                     | ✅                                                                 |
| LLVM                   | ⚠️ (scalar only; simulated `atomic<float>` and `synchronize_block` support) | ✅                                                                 |
| Remote                 | ⏩ (forwarded to underlying backends)                                                                                                                                                               | 🚧 (depends on serialization and networking)                      |

### Libraries/Applications

- 🚧 [LuisaRender](https://github.com/LuisaGroup/LuisaRender.git) (support for volumetric rendering, out-of-core tracing
  and shading, advanced sampling algorithms, custom shading language, Blender/C4D exporters, etc.)
- 🚧 [LuisaShaderToy](https://github.com/LuisaGroup/LuisaShaderToy.git) (custom shading language support planned)


### Documentation/Tutorials

- 🚧 Documentation
- ⌛ Ray tracing in one weekend with LuisaCompute
- Let me know if you have any fun idea!

