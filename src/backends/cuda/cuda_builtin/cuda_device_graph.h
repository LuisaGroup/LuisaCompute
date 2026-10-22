// CUDA device-side graph control (CUDA device runtime, CUDA 12.0+).
//
// This header is part of the concatenated device library that the CUDA backend
// prepends to every NVRTC shader (see CUDACompiler::_device_library), so it must
// stay dependency-free: no CUDA headers, no `#include`, only the self-contained
// lc_* types already declared by cuda_device_math.h.
//
// Nothing here costs anything unless it is called:
//  * the value-level primitives are inline PTX (`griddepcontrol.*` and the
//    %current_graph_exec special register) and never reference cudadevrt;
//  * the host-mutating entry points are bare `extern "C" __device__`
//    declarations of cudadevrt symbols - declaring them emits no code and no
//    undefined symbol, so a shader that never touches the device graph API keeps
//    the plain-PTX load path (see CUDAShaderNative::load_ptx).
// The LC_NVRTC_VERSION guards track the toolkit release that introduced each
// symbol, so an older toolkit never even sees the declarations.

// Self-contained scalar aliases (identical to the ones in cuda_device_math.h).
#if !defined(LC_CUDA_DEVICE_GRAPH_TYPES)
#define LC_CUDA_DEVICE_GRAPH_TYPES
using lc_uint = unsigned int;
using lc_ulong = unsigned long long;
#endif

// === Inline-PTX primitives (no cudadevrt involvement) ===

// griddepcontrol.wait: blocks the calling thread until all direct grid
// dependencies have completed (programmatic / launch-event dependencies).
__device__ inline void lc_cuda_grid_dependency_synchronize() noexcept {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// griddepcontrol.launch_dependents: allows dependent grids to be scheduled.
// This is a scheduling hint only and carries no memory visibility guarantee.
__device__ inline void lc_cuda_trigger_programmatic_launch_completion() noexcept {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// %current_graph_exec: the cudaGraphExec_t handle of the graph execution that
// launched the current grid; zero when not running inside a device graph.
[[nodiscard]] __device__ inline lc_ulong lc_cuda_get_current_graph_exec() noexcept {
    lc_ulong current_graph_exec;
    asm("mov.u64 %0, %%current_graph_exec;" : "=l"(current_graph_exec));
    return current_graph_exec;
}

// === cudadevrt entry points ===
//
// The entry points below are NOT declared here: the CUDA backend always compiles
// device code with the NVRTC `-ewp` option (see CUDADevice::create_shader), which
// makes the frontend inject the canonical declarations and scalar/typedef names
// from cuda_device_runtime_api.h (cudaError_t, cudaGraphExec_t, cudaStream_t,
// cudaGraphConditionalHandle, cudaGraphDeviceNode_t, dim3, ...). Re-declaring
// them with home-grown signatures collides with those injected declarations
// ("more than one instance of overloaded function ... has C linkage"), so the
// wrappers below call the injected functions directly with ABI-faithful casts.
//
// The symbols live in the CUDA device runtime library, which the CUDA backend
// links (cuLinkAddData with CU_JIT_INPUT_LIBRARY) whenever a shader references
// them - see the device-runtime trigger in CUDAShaderNative::load_ptx. Kernels
// that never call these wrappers emit no reference and keep the plain-PTX load
// path.
//
// ABI notes (verified against cudadevrt / the injected declarations):
//  - cudaGraphExec_t / cudaStream_t / cudaGraphDeviceNode_t are pointer-sized
//    opaque handles, passed as .b64 parameters; lc_ulong casts are exact.
//  - cudaGraphConditionalHandle is a cuuint64_t passed by value (.b64).
//  - cudaGraphKernelNodeSetGridDim takes a dim3 BY VALUE, lowered to a single
//    `.param .align 4 .b8 [12]` argument (not three .b32s); building a real
//    dim3 keeps the aggregate ABI intact.
//  - cudaError_t returns as .b32 and is discarded here.

#if LC_NVRTC_VERSION >= 12030

// cudaGraphSetConditional(cudaGraphConditionalHandle, unsigned int) -> void
// Sets the condition value associated with the conditional node owning `handle`.
__device__ inline void lc_cuda_graph_set_conditional(lc_ulong handle, lc_uint value) noexcept {
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(handle), value);
}

#endif// LC_NVRTC_VERSION >= 12030

#if LC_NVRTC_VERSION >= 12000

// cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) -> cudaError_t
// Device-side graph launch. The host API expresses the launch target as a
// special cudaStream_t constant (cuda.h); the device runtime expects that same
// constant value, so the mode is mapped here instead of at every call site.
__device__ inline void lc_cuda_graph_launch(lc_ulong graph_exec, lc_uint mode) noexcept {
    lc_ulong stream;
    switch (mode) {
        case 0u: stream = 0x0200000000000000ull; break;// cudaStreamGraphFireAndForget
        case 1u: stream = 0x0100000000000000ull; break;// cudaStreamGraphTailLaunch
        default: stream = 0x0300000000000000ull; break;// cudaStreamGraphSiblingLaunch
    }
    (void)cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph_exec),
                          reinterpret_cast<cudaStream_t>(stream));
}

#endif// LC_NVRTC_VERSION >= 12000

#if LC_NVRTC_VERSION >= 12040

// cudaGraphKernelNodeSetParam(cudaGraphDeviceNode_t, size_t, const void *, size_t)
// `value` is a 64-bit device address of the source data.
__device__ inline void lc_cuda_graph_node_set_param(
    lc_ulong node, lc_ulong offset, lc_ulong value, lc_ulong size) noexcept {
    (void)cudaGraphKernelNodeSetParam(
        reinterpret_cast<cudaGraphDeviceNode_t>(node),
        static_cast<size_t>(offset), reinterpret_cast<const void *>(value),
        static_cast<size_t>(size));
}

// cudaGraphKernelNodeSetEnabled(cudaGraphDeviceNode_t, bool)
__device__ inline void lc_cuda_graph_node_set_enabled(lc_ulong node, bool enabled) noexcept {
    (void)cudaGraphKernelNodeSetEnabled(reinterpret_cast<cudaGraphDeviceNode_t>(node), enabled);
}

// cudaGraphKernelNodeSetGridDim(cudaGraphDeviceNode_t, dim3)
__device__ inline void lc_cuda_graph_node_set_grid_dim(
    lc_ulong node, lc_uint x, lc_uint y, lc_uint z) noexcept {
    (void)cudaGraphKernelNodeSetGridDim(
        reinterpret_cast<cudaGraphDeviceNode_t>(node), dim3{x, y, z});
}

#endif// LC_NVRTC_VERSION >= 12040
