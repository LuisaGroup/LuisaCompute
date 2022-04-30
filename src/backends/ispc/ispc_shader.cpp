//
// Created by Mike Smith on 2022/2/11.
//

#include <fstream>

#include <core/clock.h>
#include <runtime/context.h>
#include <backends/ispc/ispc_codegen.h>
#include <backends/ispc/ispc_shader.h>

#ifdef LUISA_COMPUTE_ISPC_LLVM_JIT
#include <backends/ispc/ispc_jit_module.h>
#else
#include <backends/ispc/ispc_dll_module.h>
#endif

namespace luisa::compute::ispc {

ISPCShader::ISPCShader(const Context &ctx, Function func) noexcept {

    Codegen::Scratch scratch;
    ISPCCodegen codegen{scratch};
    codegen.emit(func);

    std::cout << "================= ISPC ===================\n";
    std::cout << scratch.view() << "\n";
    std::cout << "==========================================\n";


    auto name = fmt::format("func_{:016x}", func.hash());
    auto cache_dir_str = ctx.cache_directory().string();
    auto source_path = ctx.cache_directory() / fmt::format("{}.ispc", name);

    // compile
#ifdef LUISA_PLATFORM_WINDOWS
    auto ispc_exe = ctx.runtime_directory() / "ispc.exe";
#else
    auto ispc_exe = ctx.runtime_directory() / "ispc";
#endif

#ifdef LUISA_COMPUTE_ISPC_LLVM_JIT
    auto emit_opt = "--emit-llvm";
    auto object_ext = "bc";
    auto load_module = [&ctx](const std::filesystem::path &obj_path) noexcept {
        return ISPCJITModule::load(ctx, obj_path);
    };
#else
    auto emit_opt = "--emit-obj";
#ifdef LUISA_PLATFORM_WINDOWS
    auto object_ext = "obj";
#else
    auto object_ext = "o";
#endif
    auto load_module = [&ctx](const std::filesystem::path &obj_path) noexcept {
        return ISPCDLLModule::load(ctx, obj_path);
    };
#endif

    // options
    auto include_opt = fmt::format(
        "-I\"{}\"", ctx.runtime_directory().string());
    std::array ispc_options {
        "-woff",
            "--addressing=32",
#ifndef NDEBUG
            "-g",
            "-DLUISA_DEBUG",
#else
            "--opt=disable-assertions",
#endif
            "-O3",
            "--math-lib=fast",
            "--opt=fast-masked-vload",
            "--opt=fast-math",
#if defined(LUISA_PLATFORM_APPLE) && defined(__aarch64__)
            "--cpu=apple-a14",
            "--arch=aarch64",
#else
            "--cpu=core-avx2",
            "--arch=x86-64",
#endif
            emit_opt,
            include_opt.c_str(),
            func.raytracing() ? "-DLC_ISPC_RAYTRACING" : "-DLC_ISPC_NO_RAYTRACING"
    };
    luisa::string ispc_opt_string{ispc_options.front()};
    for (auto o : luisa::span{ispc_options}.subspan(1u)) {
        ispc_opt_string.append(" ").append(o);
    }

    auto object_path = ctx.cache_directory() / fmt::format("{}.{}", name, object_ext);

    // compile: write source
    {
        std::ofstream src_file{source_path};
        src_file << scratch.view();
    }

    // compile: generate object
    auto command = fmt::format(
        R"({} {} "{}" -o "{}")",
        ispc_exe.string(),
        ispc_opt_string,
        source_path.string(),
        object_path.string());
    LUISA_INFO("Compiling ISPC kernel: {}", command);

    Clock clock;
    if (auto ret = system(command.c_str()); ret != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to compile ISPC kernel. "
            "Return code: {}.",
            ret);
    }

    // load module
    _module = load_module(object_path);
    LUISA_INFO("Created ISPC shader in {} ms.", clock.toc());

    // arguments
    _argument_offsets.reserve(func.arguments().size());
    for (auto &&arg : func.arguments()) {
        auto aligned_offset = (_argument_buffer_size + 15u) / 16u * 16u;
        _argument_offsets.emplace(arg.uid(), aligned_offset);
        if (arg.type()->is_buffer()) {
            _argument_buffer_size = aligned_offset + ISPCCodegen::buffer_handle_size;
        } else if (arg.type()->is_texture()) {
            _argument_buffer_size = aligned_offset + ISPCCodegen::texture_handle_size;
        } else if (arg.type()->is_accel()) {
            _argument_buffer_size = aligned_offset + ISPCCodegen::accel_handle_size;
        } else if (arg.type()->is_bindless_array()) {
            _argument_buffer_size = aligned_offset + ISPCCodegen::bindless_array_handle_size;
        } else {
            _argument_buffer_size = aligned_offset + arg.type()->size();
        }
    }
    _argument_buffer_size = (_argument_buffer_size + 15u) / 16u * 16u;
}

size_t ISPCShader::argument_offset(uint uid) const noexcept {
    if (auto iter = _argument_offsets.find(uid);
        iter != _argument_offsets.cend()) [[likely]] {
        return iter->second;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid argument uid {}.", uid);
}

}// namespace luisa::compute::ispc
