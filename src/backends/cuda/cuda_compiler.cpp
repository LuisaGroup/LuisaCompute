//
// Created by Mike on 2021/11/8.
//

#include <fstream>

#include <backends/cuda/cuda_error.h>
#include <backends/cuda/cuda_codegen.h>
#include <backends/cuda/cuda_compiler.h>

namespace luisa::compute::cuda {

#include <backends/cuda/cuda_device_math_embedded.inl.h>
#include <backends/cuda/cuda_device_resource_embedded.inl.h>

luisa::string CUDACompiler::compile(const Context &ctx, Function function, uint32_t sm) noexcept {

    static const auto cuda_device_library_hash = hash64(cuda_device_math_source, hash64(cuda_device_resource_source));
    auto hash = hash64(sm, hash64(function.hash(), cuda_device_library_hash));

    // try memory cache
    {
        std::scoped_lock lock{_mutex};
        if (auto timepoint_iter = _function_hash_to_timepoint.find(hash);
            timepoint_iter != _function_hash_to_timepoint.end()) {
            auto old_timepoint = timepoint_iter->second;
            auto new_timepoint = ++_current_timepoint;
            timepoint_iter->second = new_timepoint;
            auto ptx_iter = _timepoint_to_ptx_and_hash.find(old_timepoint);
            auto ptx = std::move(ptx_iter->second.first);
            _timepoint_to_ptx_and_hash.erase(ptx_iter);
            _timepoint_to_ptx_and_hash.emplace(new_timepoint, std::make_pair(ptx, hash));
            return ptx;
        }
    }

    auto update_memory_cache = [hash, this](const luisa::string &ptx_string) noexcept {
        std::scoped_lock lock{_mutex};
        // another thread has updated the cache, just update timepoint
        if (auto timepoint_iter = _function_hash_to_timepoint.find(hash);
            timepoint_iter != _function_hash_to_timepoint.end()) {
            auto old_timepoint = timepoint_iter->second;
            auto new_timepoint = ++_current_timepoint;
            timepoint_iter->second = new_timepoint;
            auto ptx_iter = _timepoint_to_ptx_and_hash.find(old_timepoint);
            auto ptx = std::move(ptx_iter->second.first);
            _timepoint_to_ptx_and_hash.erase(ptx_iter);
            _timepoint_to_ptx_and_hash.emplace(new_timepoint, std::make_pair(std::move(ptx), hash));
            return;
        }
        // remove the least recently used item if cache exceeds the limit
        if (_function_hash_to_timepoint.size() >= max_cache_item_count) {
            auto lru_iter = _timepoint_to_ptx_and_hash.begin();
            auto lru_timepoint = lru_iter->first;
            auto lru_hash = lru_iter->second.second;
            _timepoint_to_ptx_and_hash.erase(lru_iter);
            _function_hash_to_timepoint.erase(lru_hash);
        }
        // emplace the new item
        auto timepoint = ++_current_timepoint;
        _function_hash_to_timepoint.emplace(hash, timepoint);
        _timepoint_to_ptx_and_hash.emplace(timepoint, std::make_pair(ptx_string, hash));
    };

    // try disk cache
    {

    }

    static thread_local Codegen::Scratch scratch;
    scratch.clear();
    CUDACodegen{scratch}.emit(function);

    auto source = scratch.view();
    LUISA_INFO("Source:\n{}", source);

    {// dump file
        std::ofstream dump{ctx.cache_directory() / fmt::format("kernel_{:016X}.cu", function.hash())};
        dump << source;
    }

    std::array header_names{"device_math.h", "device_resource.h"};
    std::array header_sources{cuda_device_math_source, cuda_device_resource_source};
    nvrtcProgram prog;
    LUISA_CHECK_NVRTC(nvrtcCreateProgram(
        &prog, source.data(), "my_kernel.cu",
        header_sources.size(), header_sources.data(), header_names.data()));
    auto sm_option = fmt::format("-arch=compute_{}", sm);
    std::array options{
        sm_option.c_str(),
        "--std=c++17",
        "--use_fast_math",
        "-default-device",
        "-restrict",
        "-include=device_math.h",
        "-include=device_resource.h",
        "-extra-device-vectorization",
        "-dw"
    };
    auto error = nvrtcCompileProgram(prog, options.size(), options.data());
    size_t log_size;
    LUISA_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1u) {
        luisa::string log;
        log.resize(log_size - 1);
        LUISA_CHECK_NVRTC(nvrtcGetProgramLog(prog, log.data()));
        LUISA_INFO("Compile log:\n{}", log);
    }
    LUISA_CHECK_NVRTC(error);

    size_t ptx_size;
    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    luisa::string ptx;
    ptx.resize(ptx_size - 1);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
    LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));

    update_memory_cache(ptx);

    return ptx;
}

CUDACompiler &CUDACompiler::instance() noexcept {
    static CUDACompiler compiler;
    return compiler;
}

}// namespace luisa::compute::cuda
