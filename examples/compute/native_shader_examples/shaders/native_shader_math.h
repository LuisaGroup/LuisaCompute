// Shared helper of the sample native shaders, included through the dispatch
// document's `config.include_dirs` (or through the source file's own directory
// for the `FilePath` route). It is plain HLSL / GLSL 450 / CUDA C++ at once.
//
// NVRTC's JIT mode only allows execution-space-annotated functions, so under
// `__CUDACC__` the helper is marked `__host__ __device__` as well.
#if defined(__CUDACC__)
#define LUISA_NATIVE_SHADER_INLINE __host__ __device__ __forceinline__
#else
#define LUISA_NATIVE_SHADER_INLINE
#endif

LUISA_NATIVE_SHADER_INLINE float luisa_native_shader_transform(float x, float k, float c) {
    return x * k + c;
}

#undef LUISA_NATIVE_SHADER_INLINE
