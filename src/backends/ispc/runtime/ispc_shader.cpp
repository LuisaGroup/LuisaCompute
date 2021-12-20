#pragma vengine_package ispc_vsproject

#include <backends/ispc/runtime/ispc_shader.h>
#include <backends/ispc/runtime/ispc_codegen.h>
#include <vstl/MD5.h>

#ifdef LUISA_COMPUTE_ISPC_LLVM_JIT
#include <backends/ispc/runtime/ispc_jit_module.h>
#else
#include <backends/ispc/runtime/ispc_dll_module.h>
#endif

namespace lc::ispc {

Shader::~Shader() = default;

Shader::Shader(
    const Context &ctx, Function func)
    : func(func) {

    // generate code
    luisa::string source;
    CodegenUtility::PrintFunction(func, source, func.block_size());

    // TODO: cache
    auto name = fmt::format("func_{:016x}", func.hash());
    auto cache_dir_str = ctx.cache_directory().string();
    auto source_path = ctx.cache_directory() / fmt::format("{}.ispc", name);

    // compile
#ifdef LUISA_PLATFORM_WINDOWS
    auto ispc_exe = ctx.runtime_directory() / "backends" / "ispc_support" / "ispc.exe";
#else
    auto ispc_exe = ctx.runtime_directory() / "backends" / "ispc_support" / "ispc";
#endif

#ifdef LUISA_COMPUTE_ISPC_LLVM_JIT
    auto emit_opt = "--emit-llvm";
    auto object_ext = "bc";
    auto load_module = [&ctx](const std::filesystem::path &obj_path) noexcept {
        return JITModule::load(ctx, obj_path);
    };
#else
    auto emit_opt = "--emit-obj";
    auto object_ext = "obj";
    auto load_module = [&ctx](const std::filesystem::path &obj_path) noexcept {
        return DLLModule::load(ctx, obj_path);
    };
#endif

    // options
    auto include_opt = fmt::format(
        "-I\"{}\"",
        std::filesystem::canonical(
            ctx.runtime_directory() / "backends" / "ispc_support")
            .string());
    std::array ispc_options{
        "-woff",
        "-O3",
        "--math-lib=fast",
        "--opt=fast-masked-vload",
        "--opt=fast-math",
        "--opt=force-aligned-memory",
#ifndef __aarch64__
        "--cpu=core-avx2",
#else
        "--cpu=apple-a14",
#endif
        "--enable-llvm-intrinsics",
        emit_opt,
        include_opt.c_str()};
    luisa::string ispc_opt_string{ispc_options.front()};
    for (auto o : std::span{ispc_options}.subspan(1u)) {
        ispc_opt_string.append(" ").append(o);
    }

    auto object_path = ctx.cache_directory() / fmt::format("{}.{}", name, object_ext);

    // compile: write source
    {
        std::ofstream src_file{source_path};
        src_file << source;
    }

    // compile: generate object
    auto command = fmt::format(
        R"({} {} "{}" -o "{}")",
        ispc_exe.string(),
        ispc_opt_string,
        source_path.string(),
        object_path.string());
    LUISA_INFO("Compiling ISPC kernel: {}", command);
    if (auto ret = system(command.c_str()); ret != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to compile ISPC kernel. "
            "Return code: {}.",
            ret);
    }

    // load module
    executable = load_module(object_path);

    // arguments
    size_t sz = 0;
    for (auto &&i : func.arguments()) {
        varIdToArg.Emplace(i.uid(), sz);
        sz++;
    }
}
size_t Shader::GetArgIndex(uint varID) const {
    auto ite = varIdToArg.Find(varID);
    if (!ite) return std::numeric_limits<size_t>::max();
    return ite.Value();
}

ThreadTaskHandle Shader::dispatch(
    ThreadPool *tPool,
    uint3 sz,
    ArgVector vec) const {
    auto blockSize = func.block_size();
    auto blockCount = (sz + blockSize - 1u) / blockSize;
    auto totalCount = blockCount.x * blockCount.y * blockCount.z;
    auto handle = tPool->GetParallelTask(
        [=, vec = std::move(vec)](size_t i) noexcept {
            uint blockIdxZ = i / (blockCount.y * blockCount.x);
            i -= blockCount.y * blockCount.x * blockIdxZ;
            uint blockIdxY = i / blockCount.x;
            i -= blockIdxY * blockCount.x;
            uint blockIdxX = i;
            executable->invoke(blockCount, make_uint3(blockIdxX, blockIdxY, blockIdxZ), sz, vec.data());
        },
        totalCount,
        true);
    return handle;
}
}// namespace lc::ispc
