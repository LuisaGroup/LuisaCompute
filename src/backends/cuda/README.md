# OptiX host API

The CUDA backend supports the official [NVIDIA OptiX headers](https://github.com/NVIDIA/optix-dev)
from **OptiX 9.0 and later**, including 9.0 (ABI 105) and 9.1 (ABI 118).
By default it uses the `src/ext/optix` submodule, pinned to **v9.1.0**, commit
`f1f6dd803f3159992d248178f6e09421c6eb8b6d`.
Initialize the default headers with the other dependencies, or separately:

```sh
git submodule update --init src/ext/optix
```

CMake users can select another SDK without changing that submodule:

```sh
cmake -S . -B build -DLUISA_COMPUTE_OPTIX_INCLUDE_DIR="/path/to/OptiX-9.0/include"
```

The path must contain `optix.h`, `optix_host.h`, `optix_types.h`, and
`optix_function_table.h` from the same SDK. Invalid paths and versions below
9.0 are configuration errors; they do not silently select the bundled SDK.
Set the option to an empty string to restore the bundled headers.
A parent project can also set the variable before adding LuisaCompute:

```cmake
set(LUISA_COMPUTE_OPTIX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/optix/include")
add_subdirectory(LuisaCompute)
```

Both CMake and xmake expose these headers only to the CUDA backend's host
compilation. `optix_api.h` is a small loader interface, not a private copy of
the SDK declarations. `optix_api.cpp` retains runtime loading of the driver
library (`nvoptix.dll` / `libnvoptix.so.1`), platform-specific search paths,
and the non-fatal `available()` probe used by software ray-tracing fallback.
There is no link-time OptiX runtime dependency.

The selected SDK determines the required driver ABI: [OptiX 9.0](https://github.com/NVIDIA/optix-dev/releases/tag/v9.0.0)
requires **NVIDIA R570 or newer**, while [OptiX 9.1](https://github.com/NVIDIA/optix-dev/releases/tag/v9.1.0)
requires **R590 or newer**. SDK selection happens at build time; the loader
requests `OPTIX_ABI_VERSION` from those headers. It does not negotiate an older
SDK ABI at runtime. A driver that cannot provide the selected ABI fails the
capability probe and uses the existing software ray-tracing fallback. This does not emulate OptiX
features such as its denoiser, which still require a compatible OptiX runtime.
The SDK removed displacement micromap support in 9.1; LuisaCompute does not
use those removed host APIs or device intrinsics.

The pipeline links no miss program groups: a ray that hits nothing simply ends
traversal, which is what the ray-query loop expects. `optixLaunch` nevertheless
rejects an empty miss region (`"sbt->missRecordCount" is zero`), so the single
miss slot aliases a hit-group record. `LUISA_OPTIX_VALIDATION=1` therefore
aborts with `VALIDATION_ERROR [UNEXPECTED_SHADER_TYPE]`; validation mode stays
unusable until the generated modules carry a dedicated `__miss__` program. That
message is a known limitation of this SBT layout, not a ray-tracing failure.

The lightweight OptiX intrinsics and layouts in `cuda_builtin/` remain
handwritten. Do **not** include the official device headers in that embedded
NVRTC source bundle: this would increase runtime compilation work. Layout
assertions in `optix_api.h` guard the host/device ABI boundary. Host pipeline
options added in 9.1 are used only when the selected headers provide them;
9.0 sets traversable depth through `optixPipelineSetStackSize`.
When adopting a newer SDK, review the lightweight declarations and LLVM codegen
alongside the official ABI. The CUDA LLVM shader cache includes `OPTIX_VERSION`
so changing SDK versions does not reuse shaders compiled for the previous SDK.

The upstream SDK has its own licensing terms. See
[`LICENSE.txt`](../../ext/optix/LICENSE.txt) and
[`license_info.txt`](../../ext/optix/license_info.txt) in the pinned submodule;
LuisaCompute's license does not replace those terms.
