# OptiX host API

The CUDA backend uses the official [NVIDIA OptiX headers](https://github.com/NVIDIA/optix-dev)
from the `src/ext/optix` submodule, pinned to **v9.0.0**, commit
`fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd` (OptiX ABI 105).
Initialize it with the other dependencies, or separately:

```sh
git submodule update --init src/ext/optix
```

Both CMake and xmake expose these headers only to the CUDA backend's host
compilation. `optix_api.h` is a small loader interface, not a private copy of
the SDK declarations. `optix_api.cpp` retains runtime loading of the driver
library (`nvoptix.dll` / `libnvoptix.so.1`), platform-specific search paths,
and the non-fatal `available()` probe used by software ray-tracing fallback.
There is no link-time OptiX runtime dependency.

The lightweight OptiX intrinsics and layouts in `cuda_builtin/` remain
handwritten. Do **not** include the official device headers in that embedded
NVRTC source bundle: this would increase runtime compilation work. Version
and layout assertions in `optix_api.h` guard the host/device ABI boundary.
When upgrading the submodule, review these declarations and LLVM codegen
alongside the official ABI; updating the version assertion alone is not enough.

The upstream SDK has its own licensing terms. See
[`LICENSE.txt`](../../ext/optix/LICENSE.txt) and
[`license_info.txt`](../../ext/optix/license_info.txt) in the pinned submodule;
LuisaCompute's license does not replace those terms.
