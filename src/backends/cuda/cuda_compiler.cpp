#include <luisa/core/clock.h>
#include <luisa/core/binary_io.h>
#include "cuda_error.h"
#include "cuda_device.h"
#include "optix_api.h"
#include "cuda_builtin_embedded.h"
#include "cuda_compiler.h"

#ifndef LUISA_COMPUTE_STANDALONE_NVRTC_DLL
extern "C" {
char *luisa_nvrtc_compile_to_ptx(
    const char *filename, const char *src,
    const char *const *options, size_t num_options);
void luisa_nvrtc_free_ptx(char *ptx);
int luisa_nvrtc_version();
}
#endif

namespace luisa::compute::cuda {

luisa::string CUDACompiler::compile(const luisa::string &src, const luisa::string &src_filename,
                                    luisa::span<const char *const> options,
                                    const CUDAShaderMetadata *metadata) const noexcept {

    Clock clk;

#ifndef NDEBUG
    // in debug mode, we always recompute the hash, so
    // that we can check the hash if metadata is provided
    auto hash = compute_hash(src, options);
    if (metadata) { LUISA_ASSERT(metadata->checksum == hash, "Hash mismatch!"); }
#else
    auto hash = metadata ? metadata->checksum : compute_hash(src, options);
#endif

    if (auto ptx = _cache->fetch(hash)) { return *ptx; }

    auto filename = src_filename.empty() ? "my_kernel.cu" : src_filename.c_str();
    auto ptx_c_str = _compile_func(filename, src.c_str(), options.data(), options.size());
    auto ptx = luisa::string{ptx_c_str};
    _free_func(ptx_c_str);
    LUISA_VERBOSE("CUDACompiler::compile() took {} ms.", clk.toc());
    return ptx;
}

size_t CUDACompiler::type_size(const Type *type) noexcept {
    if (!type->is_custom()) { return type->size(); }
    // TODO: support custom types
    if (type->description() == "LC_IndirectKernelDispatch") {
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

CUDACompiler::CUDACompiler(const CUDADevice *device) noexcept
    : _device{device},
      _device_library{[] {
          luisa::string device_library;
          auto device_half = luisa::string_view{
              luisa_cuda_builtin_cuda_device_half,
              sizeof(luisa_cuda_builtin_cuda_device_half)};
          auto device_math = luisa::string_view{
              luisa_cuda_builtin_cuda_device_math,
              sizeof(luisa_cuda_builtin_cuda_device_math)};
          auto device_resource = luisa::string_view{
              luisa_cuda_builtin_cuda_device_resource,
              sizeof(luisa_cuda_builtin_cuda_device_resource)};

          device_library.resize(device_half.size() +
                                device_math.size() +
                                device_resource.size());
          std::memcpy(device_library.data(),
                      device_half.data(), device_half.size());
          std::memcpy(device_library.data() + device_half.size(),
                      device_math.data(), device_math.size());
          std::memcpy(device_library.data() + device_half.size() + device_math.size(),
                      device_resource.data(), device_resource.size());
          return device_library;
      }()},
      _cache{Cache::create(max_cache_item_count)} {

#ifdef LUISA_COMPUTE_STANDALONE_NVRTC_DLL
    _nvrtc_module = DynamicModule::load("lc-backend-cuda-nvrtc");
    LUISA_ASSERT(_nvrtc_module, "Failed to load CUDA NVRTC module.");
    _version_func = _nvrtc_module.function<nvrtc_version_func>("luisa_nvrtc_version");
    _compile_func = _nvrtc_module.function<nvrtc_compile_func>("luisa_nvrtc_compile_to_ptx");
    _free_func = _nvrtc_module.function<nvrtc_free_func>("luisa_nvrtc_free_ptx");
#else
    _version_func = &luisa_nvrtc_version;
    _compile_func = &luisa_nvrtc_compile_to_ptx;
    _free_func = &luisa_nvrtc_free_ptx;
#endif

    _nvrtc_version = _version_func();
    LUISA_VERBOSE("CUDA NVRTC version: {}.", _nvrtc_version);
    LUISA_VERBOSE("CUDA device library size = {} bytes.", _device_library.size());
}

uint64_t CUDACompiler::compute_hash(const string &src,
                                    luisa::span<const char *const> options) const noexcept {
    auto hash = hash_value(src);
    for (auto o : options) { hash = hash_value(o, hash); }
    return hash;
}

}// namespace luisa::compute::cuda
