# OptiX host API

The CUDA backend uses the official [NVIDIA OptiX headers](https://github.com/NVIDIA/optix-dev)
from the `src/ext/optix` submodule, pinned to **v9.1.0**, commit
`f1f6dd803f3159992d248178f6e09421c6eb8b6d` (OptiX ABI 118).
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

OptiX 9.1 requires an **NVIDIA R590 or newer driver**, as documented in the
[upstream release](https://github.com/NVIDIA/optix-dev/releases/tag/v9.1.0).
An older driver that cannot provide ABI 118 fails the capability probe and
uses the existing software ray-tracing fallback. This does not emulate OptiX
features such as its denoiser, which still require a compatible OptiX runtime.
The SDK removed displacement micromap support in 9.1; LuisaCompute does not
use those removed host APIs or device intrinsics.

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
