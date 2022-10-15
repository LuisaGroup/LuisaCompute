# LuisaCompute

![teaser](https://user-images.githubusercontent.com/7614925/195987646-fe932ebe-ca6e-4d6e-ab2a-203bcfd3d559.jpg)

LuisaCompute is a high-performance cross-platform computing framework for graphics and beyond.

LuisaCompute is also the *rendering framework* described in the **SIGGRAPH Asia 2022** paper
> ***LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures***.

See also [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for the *rendering application* as described in the paper; and please visit the [project page](https://luisa-render.com) for other information about the paper and the project.

## Overview

LuisaCompute seeks to balance the seemingly ever-conflicting pursuits for ***unification***, ***programmability***, and ***performance***. To achieve this goal, we design three major components:
- A DSL embedded inside modern C++ for kernel programming exploiting JIT code generation and compilation;
- A unified runtime for cross-platform resource management and command scheduling; and
- Multiple optimized backends, including CUDA, DirectX, Metal, LLVM, and ISPC.

To demonstrate the practicality of the system, we build a Monte Carlo renderer, [LuisaRender](https://github.com/LuisaGroup/LuisaRender), atop the framework, which is 5–11× faster than [PBRT-v4](https://github.com/mmp/pbrt-v4) and 4–16× faster than [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) on modern GPUs.

## Building

> Note: LuisaCompute is a *rendering framework* rather than a *renderer* itself. It is design to provide general computation functionalities on modern stream-processing hardware, on which high-performance, cross-platform graphics applications can be easily built. If you would like to just try out a Monte Carlo renderer out of the box rather than building one from the scratch, please see [LuisaRender](https://github.com/LuisaGroup/LuisaRender).

LuisaCompute follows the standard CMake build process. Basically these steps:

- Check your hardware and platform. Currently, we support CUDA on Linux and Windows; DirectX on Windows; Metal on macOS; and ISPC and LLVM on all the major platforms. For CUDA and DirectX, an RTX-enabled graphics card, e.g., NVIDIA RTX 20 and 30 series, is required.

- Prepare the environment and dependencies. We recommend using the latest IDEs, Compilers, CMake, CUDA drivers, etc. Since we aggressively use new technologies like C++20 and OptiX 7.1+, you may need to, for example, upgrade your VS to 2019 or 2022, and install CUDA 11.0+. Note that if you would like to enable the CUDA backend, [OptiX](https://developer.nvidia.com/designworks/optix/download) is required. For some tests like the toy path tracer, [OpenCV](opencv.org) is also required.

- Clone the repo with the `--recursive` option, e.g., `git clone --recursive https://github.com/LuisaGroup/LuisaCompute.git`. We use Git submodules to manage most dependencies, so a `--recursive` clone is required. Also, as we are not allowed to provide the OptiX headers in tree, you have to copy them from `<optix-installation>/include` to `src/backends/cuda/optix`, so that the latter folder *directly* contains `optix.h`. We applogize for this inconvenience.

- Configure the project using CMake. E.g., for command line, `cd` into the project folder and type `cmake -S . -B <build-folder>`. You might also want to specify your favorite generators and build types using options like `-G Ninja` and `-D CMAKE_BUILD_TYPE=Release`. A typical, full command sequence for this would be like
```bash
cd LuisaCompute
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
```

- If the configuration succeeds, you are now able to build the project. Type `cmake --build <build-folder>` in the command line, or push the build button if you generated, e.g., a VS project. (And in case the configuration step unluckily failed :-(, please file an [issue](https://github.com/LuisaGroup/LuisaCompute/issues)). 

- After building, you will find the binaries under `<build-folder>/bin`. You can now play around with them, for example type `<build-folder>/bin/test_texture_io cuda` to generate a gradient color texture with the CUDA backend. 

See also [BUILD.md](BUILD.md) for details on platform requirements, configuration options, and other precautions.

## Usage



## Applications

We implement several proof-of-concept examples in tree under `src/examples`. Besides, you may also found the following applications interesting:

- [LuisaRender](https://github.com/LuisaGroup/LuisaRender.git), a high-performance cross-platform Monte Carlo renderer.
- [LuisaShaderToy](https://github.com/LuisaGroup/LuisaShaderToy.git), a collection of amazing shaders ported from [Shadertoy](https://www.shadertoy.com).

## Documentation/Tutorials

Sorry that we are still working on them. Currently, we would recommand you to read the original [paper](https://luisa-render.com) and learn the basic usages through the examples.

If you have any problem or suggestion, please just feel free to open an [issue](https://github.com/LuisaGroup/LuisaCompute/issues) or start a [discussion](https://github.com/LuisaGroup/LuisaCompute/discussions). We are very happy to hear from you!

## Citation

```bibtex
@article{Zheng2022LuisaRender,
    title = {LuisaRender: A High-Performance Rendering Framework with Layered and Unified Interfaces on Stream Architectures},
    author = {Zheng, Shaokun and Zhou, Zhiqian and Chen, Xin and Yan, Difei and Zhang, Chuyan and Geng, Yuefeng and Gu, Yan and Xu, Kun},
    journal = {ACM Trans. Graph.},
    volume = {41},
    number = {6},
    year = {2022},
    issue_date = {December 2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3550454.3555463},
    doi = {10.1145/3550454.3555463},
    month = {dec},
    articleno = {232},
    numpages = {19}
}
```

The [publisher version](https://doi.org/10.1145/3550454.3555463) of the paper should be available soon before SIGGRAPH Asia 2022.
