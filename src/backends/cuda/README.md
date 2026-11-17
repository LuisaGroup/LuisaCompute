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

The pipeline links no miss program groups: a ray that hits nothing simply ends
traversal, which is what the ray-query loop expects. `optixLaunch` nevertheless
rejects an empty miss region (`"sbt->missRecordCount" is zero`), so the single
miss slot aliases a hit-group record. `LUISA_OPTIX_VALIDATION=1` therefore
aborts with `VALIDATION_ERROR [UNEXPECTED_SHADER_TYPE]`; validation mode stays
unusable until the generated modules carry a dedicated `__miss__` program. That
message is a known limitation of this SBT layout, not a ray-tracing failure.

The lightweight OptiX intrinsics and layouts in `cuda_builtin/` remain
handwritten. Do **not** include the official device headers in that embedded
NVRTC source bundle: this would increase runtime compilation work. Version
and layout assertions in `optix_api.h` guard the host/device ABI boundary.
When upgrading the submodule, review these declarations and LLVM codegen
alongside the official ABI; updating the version assertion alone is not enough.

The upstream SDK has its own licensing terms. See
LICENSE.txt (../../ext/optix/LICENSE.txt) and
license_info.txt (../../ext/optix/license_info.txt) in the pinned submodule;
LuisaCompute's license does not replace those terms.

Programmatic dependent launch (stream form)

A CUDA stream serializes its grids: a kernel does not start before its
predecessor retired, which leaves a launch/flush bubble at every kernel->kernel
boundary. CUDA 12.0 exposes `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
for exactly that bubble: a grid launched with the attribute may start while the
previous kernel of the stream is still running, provided it waits for its
programmatic dependency (`griddepcontrol.wait`) before reading what that kernel
wrote. This is *not* a dependency system and it does not batch commands — it is a
single pairwise relaxation, and the backend never uses it unless a wait is in
place.

The feature has two halves, decided at different times:

* **Codegen (per device, at shader compile time).**
  `CUDADeviceConfigExt::enable_programmatic_stream_serialization()` (default
  `false`) makes the backend compile `lc_cuda_grid_dependency_synchronize();` as
  the first statement of `kernel_main`, so every DSL compute kernel of that
  device waits at its entry and *any* kernel->kernel pair becomes safe without
  dependency analysis. The flag ends up in the shader metadata
  (`requires_grid_dependency_synchronize`, which already existed for kernels that
  call `grid_dependency_synchronize()` themselves), it changes the generated
  source and therefore the shader cache key, and it is not emitted for the OptiX
  raygen entry or for the LLVM codegen route.
* **Launch policy (per launch, at submit time).** `cuda_pdl_launch.h` decides
  between `cuLaunchKernelEx` (attribute set) and the unchanged `cuLaunchKernel`.
  A launch is relaxed only when the shader waits **and** the immediately previous
  operation on the stream was a kernel launch (`CUDAStream` tracks that; a copy,
  a build, an event record/wait, a host callback or a fresh stream is a hard
  boundary) **and** the policy is on **and** the device is compute capability 9.0+
  with driver 12.0+.

Runtime control and observability are in
`include/luisa/backends/ext/cuda/cuda_pdl_ext.h`:
`Device::extension<CUDAPdlStreamExt>()` exposes `pdl_supported()`,
`auto_prologue_enabled()`, the `pdl_stream_enabled()`/`set_pdl_stream_enabled()`
policy switch, and per-device launch counters (relaxed / strict / driver
fallbacks) for A/B runs. The policy is **on** by default — relaxing a kernel that
already waits is what this backend did before the prologue existed — while the
prologue is the opt-in part, because it changes the generated code of every
kernel. `LUISA_DISABLE_PDL_STREAM=1` forces both halves off for the whole process
and reproduces the pre-change generated source byte for byte.

Below compute capability 9.0 the feature is inert by construction:
`cuda_builtin/cuda_device_graph.h` compiles `griddepcontrol.*` away (an unguarded
`asm` is accepted by NVRTC for an older `-arch` and then rejected by the driver at
module load with `CUDA_ERROR_INVALID_PTX`), the wait becomes a no-op, and the
launch attribute is never requested. Coverage: `src/tests/unit/core/test_cuda_pdl_launch_predicate.cpp`
(the decision matrix, host only) and `src/tests/integration/runtime/test_cuda_stream_pdl.cpp`
(host -> device: a chained producer/consumer workload, the launch counters, the
non-kernel boundary and an A/B against a device without the prologue). Whether a
relaxed grid really overlaps can only be measured on sm_90+ hardware.
