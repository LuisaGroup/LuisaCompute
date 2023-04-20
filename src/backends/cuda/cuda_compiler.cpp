//
// Created by Mike on 2021/11/8.
//

#include <fstream>

#include <core/clock.h>
#include <core/binary_io.h>
#include <runtime/context_paths.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/optix_api.h>
#include <backends/cuda/cuda_builtin_embedded.h>
#include <backends/cuda/cuda_compiler.h>

namespace luisa::compute::cuda {

luisa::string CUDACompiler::compile(const luisa::string &src,
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

    std::array header_names{"device_library.h"};
    std::array header_sources{_device_library.c_str()};
    nvrtcProgram prog;
    LUISA_CHECK_NVRTC(nvrtcCreateProgram(
        &prog, src.data(), "my_kernel.cu",
        header_sources.size(), header_sources.data(), header_names.data()));
    auto error = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());
    size_t log_size;
    LUISA_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1u) {
        luisa::string log;
        log.resize(log_size - 1);
        LUISA_CHECK_NVRTC(nvrtcGetProgramLog(prog, log.data()));
        LUISA_WARNING_WITH_LOCATION("Compile log:\n{}", log);
    }
    LUISA_CHECK_NVRTC(error);
    size_t ptx_size;
    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    luisa::string ptx;
    if (metadata) {
        ptx.append(luisa::format(
            "// METADATA: {}\n\n",
            serialize_cuda_shader_metadata(*metadata)));
    }
    auto header_size = ptx.size();
    ptx.resize(header_size + ptx_size - 1u);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data() + header_size));
    LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    LUISA_VERBOSE_WITH_LOCATION("CUDACompiler::compile() took {} ms.", clk.toc());
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
      _nvrtc_version{[] {
          auto ver_major = 0;
          auto ver_minor = 0;
          LUISA_CHECK_NVRTC(nvrtcVersion(&ver_major, &ver_minor));
          return static_cast<uint>(ver_major * 10000 + ver_minor * 100);
      }()},
      _device_library{[] {
          luisa::string device_library;
          auto device_math = luisa::string_view{
              luisa_cuda_builtin_cuda_device_math,
              sizeof(luisa_cuda_builtin_cuda_device_math)};
          auto device_resource = luisa::string_view{
              luisa_cuda_builtin_cuda_device_resource,
              sizeof(luisa_cuda_builtin_cuda_device_resource)};

          device_library.resize(device_math.size() + device_resource.size());
          std::memcpy(device_library.data(),
                      device_math.data(), device_math.size());
          std::memcpy(device_library.data() + device_math.size(),
                      device_resource.data(), device_resource.size());
          return device_library;
      }()},
      _library_hash{hash_value(_device_library)},
      _cache{Cache::create(max_cache_item_count)} {
    LUISA_VERBOSE_WITH_LOCATION("CUDA NVRTC version: {}.", _nvrtc_version);
    LUISA_VERBOSE_WITH_LOCATION("CUDA device library size = {} bytes, hash = {:016X}.",
                                _device_library.size(), _library_hash);
}

uint64_t CUDACompiler::compute_hash(const string &src, luisa::span<const char *const> options) const noexcept {
    auto hash = hash_value(src, _library_hash);
    for (auto o : options) { hash = hash_value(o, hash); }
    return hash;
}

}// namespace luisa::compute::cuda
