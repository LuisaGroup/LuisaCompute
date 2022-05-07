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

    auto hash = hash64(sm, function.hash());
    if (auto ptx = _cache->fetch(hash)) { return *ptx; }

    auto ver_major = 0;
    auto ver_minor = 0;
    LUISA_CHECK_NVRTC(nvrtcVersion(&ver_major, &ver_minor));
    static const auto library_hash = hash64(
        cuda_device_math_source,
        hash64(cuda_device_resource_source,
               ver_major * 1000 + ver_minor * 10));

    auto sm_option = fmt::format("-arch=compute_{}", sm);
    auto rt_option = fmt::format("-DLC_RAYTRACING_KERNEL={}", function.raytracing());
    std::array options{
        sm_option.c_str(),
        rt_option.c_str(),
        "--std=c++17",
        "--use_fast_math",
        "-default-device",
        "-restrict",
        "-include=device_math.h",
        "-include=device_resource.h",
        "-extra-device-vectorization",
        "-dw",
        "-w"};

    auto opt_hash = Hash64::default_seed;
    for (auto o : options) { opt_hash = hash64(o, opt_hash); }

    auto file_name = fmt::format(
        "func_{:016x}.lib_{:016x}.opt_{:016x}",
        function.hash(), library_hash, opt_hash);
    auto &&cache_dir = ctx.cache_directory();
    auto ptx_file_name = file_name + ".ptx";
    auto cu_file_name = file_name + ".cu";
    auto ptx_file_path = cache_dir / ptx_file_name;

    // try disk cache
    if (std::ifstream ptx_file{ptx_file_path}; ptx_file.is_open()) {
        LUISA_INFO("Found compilation cache: '{}'.", ptx_file_name);
        luisa::string ptx{
            std::istreambuf_iterator<char>{ptx_file},
            std::istreambuf_iterator<char>{}};
        _cache->update(hash, ptx);
        return ptx;
    }
    LUISA_INFO(
        "Failed to load compilation cache for kernel {:016X},"
        " falling back to re-compiling.",
        function.hash());

    // compile
    static thread_local Codegen::Scratch scratch;
    scratch.clear();
    CUDACodegen{scratch}.emit(function);

    auto source = scratch.view();
    LUISA_VERBOSE_WITH_LOCATION("Generated CUDA source:\n{}", source);

    // save the source for debugging
    {
        std::ofstream cu_file{cache_dir / cu_file_name};
        cu_file << source;
    }

    std::array header_names{"device_math.h", "device_resource.h"};
    std::array header_sources{cuda_device_math_source, cuda_device_resource_source};
    nvrtcProgram prog;
    LUISA_CHECK_NVRTC(nvrtcCreateProgram(
        &prog, source.data(), "my_kernel.cu",
        header_sources.size(), header_sources.data(), header_names.data()));
    auto error = nvrtcCompileProgram(prog, options.size(), options.data());
    size_t log_size;
    LUISA_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1u) {
        luisa::string log;
        log.resize(log_size - 1);
        LUISA_CHECK_NVRTC(nvrtcGetProgramLog(prog, log.data()));
        std::cerr << "Compile log:\n" << log << std::flush;
    }
    LUISA_CHECK_NVRTC(error);

    size_t ptx_size;
    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    luisa::string ptx;
    ptx.resize(ptx_size - 1);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
    LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));

    _cache->update(hash, ptx);

    // save cache
    std::ofstream ptx_file{ptx_file_path};
    ptx_file << ptx;
    return ptx;
}

CUDACompiler &CUDACompiler::instance() noexcept {
    static CUDACompiler compiler;
    return compiler;
}

}// namespace luisa::compute::cuda
