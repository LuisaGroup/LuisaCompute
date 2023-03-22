//
// Created by Mike on 2021/11/8.
//

#include <fstream>

#include <core/clock.h>
#include <core/binary_io.h>
#include <runtime/context_paths.h>
#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_device.h>
#include <backends/cuda/cuda_compiler.h>
#include <backends/cuda/optix_api.h>

namespace luisa::compute::cuda {

luisa::string CUDACompiler::compile(const luisa::string &src,
                                    luisa::span<const char *const> options,
                                    luisa::optional<uint64_t> precomputed_hash) const noexcept {

#ifndef NDEBUG
    auto recomputed_hash = compute_hash(src, options);
    LUISA_ASSERT(!precomputed_hash || *precomputed_hash == recomputed_hash,
                 "Hash mismatch!");
    precomputed_hash.emplace(recomputed_hash);
#endif

    Clock clk;
    auto hash = precomputed_hash.value_or(compute_hash(src, options));
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
    auto ptx = checksum_header(hash).append("\n\n");
    auto checksum_header_size = ptx.size();
    size_t ptx_size;
    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(checksum_header_size + ptx_size - 1u);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data() + checksum_header_size));
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
    if (type->description() == "LC_RayQuery") {
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
      _device_library{[device] {
          luisa::string device_library;
          auto device_math_stream = device->io()->read_internal_shader("cuda_device_math.h");
          auto device_resource_stream = device->io()->read_internal_shader("cuda_device_resource.h");
          device_library.resize(device_math_stream->length() +
                                device_resource_stream->length());
          device_math_stream->read(luisa::span{
              reinterpret_cast<std::byte *>(device_library.data()),
              device_math_stream->length()});
          device_resource_stream->read(luisa::span{
              reinterpret_cast<std::byte *>(device_library.data() +
                                            device_math_stream->length()),
              device_resource_stream->length()});
#ifndef NDEBUG
          LUISA_ASSERT(device_library.size() == std::strlen(device_library.c_str()),
                       "Device library contains null characters.");
#endif
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

luisa::string CUDACompiler::checksum_header(uint64_t hash) const noexcept {
    return luisa::format("// CHECKSUM {:016X}", hash);
}

}// namespace luisa::compute::cuda
