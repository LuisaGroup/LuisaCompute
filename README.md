# LuisaCompute

LuisaCompute is a high-performance cross-platform computing framework for graphics and beyond.

LuisaCompute is also the *rendering framework* described in the **SIGGRAPH Asia 2022** paper
> ***LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures***.

See also [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for the *rendering application* as described in the paper; and please visit the [project page](https://luisa-render.com) for other information about the paper and the project.

## Building and Running

See [BUILD](BUILD.md).

## Features

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

### Backends

| Implementation         | Codegen Status                                                                                                                                                                                     | Runtime Status                                                    |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| CUDA                   | ✅                                                                                                                                                                                                  | ✅                                                                 |
| DirectX                | ✅ (simulated `atomic<float>` support)                                                                                                                                                              | ✅                                                                 |
| Metal                  | ✅ (simulated `atomic<float>` support)                                                                                                                                                              | ⚠️ (acceleration structure compaction disabled due to Metal bugs) |
| ISPC<br />[Deprecated] | ⚠️ (no support for shared memory or `synchonize_block`; compiles very slow; simulated `atomic<float>` support)                                                                                     | ✅                                                                 |
| LLVM                   | ✅ (scalar mode; simulated `atomic<float>` and `synchronize_block` support) | ✅ |

### Libraries/Applications

- 🚧 [LuisaRender](https://github.com/LuisaGroup/LuisaRender.git)
- 🚧 [LuisaShaderToy](https://github.com/LuisaGroup/LuisaShaderToy.git)


### Documentation/Tutorials

- 🚧 Documentation
- ⌛ Ray tracing in one weekend with LuisaCompute
- Let me know if you have any fun idea!

