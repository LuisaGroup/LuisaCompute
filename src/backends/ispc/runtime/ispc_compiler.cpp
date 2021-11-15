#pragma vengine_package ispc_vsproject

#include <core/platform.h>
#include <backends/ispc/runtime/ispc_compiler.h>

namespace lc::ispc {

// windows
//#if defined(LUISA_PLATFORM_WINDOWS)
//namespace detail {
//static std::string_view FOLDER_NAME = "ispc_backend\\";
//}
//static void GenerateDll(
//    std::string_view code,
//    luisa::string const &fileName,
//    luisa::string const &libName) {
//    //write text
//    luisa::string textName = fileName + ".txt";
//    {
//        auto f = fopen(textName.c_str(), "wb");
//        if (f) {
//            auto disp = vstd::create_disposer([&] { fclose(f); });
//            fwrite(code.data(), code.size(), 1, f);
//        }
//    }
//    //compile
//    luisa::string compileCmd(detail::FOLDER_NAME);
//    compileCmd << "ispc.exe -O2 "
//               << textName << " -o " << fileName << ".obj -woff";
//    system(compileCmd.c_str());
//    compileCmd.clear();
//    compileCmd << detail::FOLDER_NAME
//               << "link.exe /DLL /NOLOGO /OUT:"sv
//               << libName
//               << " /DYNAMICBASE \""
//               << detail::FOLDER_NAME
//               << "msvcrt.lib\" /NOENTRY /EXPORT:run /NODEFAULTLIB "sv
//               << fileName
//               << ".obj"sv;
//    system(compileCmd.c_str());
//    remove((fileName + ".obj").c_str());
//    //remove((fileName + ".txt").c_str());
//    remove((fileName + ".exp").c_str());
//    remove((fileName + ".lib").c_str());
//    //link
//}
//luisa::string Compiler::CompileCode(
//    const Context &ctx,
//    std::string_view code) const {
//    vstd::MD5 md5(code);
//    luisa::string fileName;
//    fileName << detail::FOLDER_NAME << md5.ToString();
//    constexpr size_t MD5_BASE64_STRLEN = 22;
//    fileName.resize(MD5_BASE64_STRLEN + detail::FOLDER_NAME.size());
//    luisa::string dllName = fileName + ".dll";
//    if (!vstd::FileSystem::IsFileExists(dllName)) {
//        GenerateDll(code, fileName, dllName);
//    }
//    return fileName;
//}
//#elif defined(LUISA_PLATFORM_UNIX)
//TODO: other platforms
JITModule Compiler::CompileCode(
    const Context &ctx,
    std::string_view code) const {
    LUISA_VERBOSE_WITH_LOCATION("Generated ISPC source:\n{}", code);
    auto ispc_exe = "ispc";
    auto include_opt = fmt::format(
        "-I\"{}\"",
        std::filesystem::canonical(
            ctx.runtime_directory() / "backends" / "ispc_device_include")
            .string());
    std::array ispc_options{
        "-woff",
        "-O3",
        "--math-lib=fast",
        "--opt=fast-masked-vload",
        "--opt=fast-math",
        "--opt=force-aligned-memory",
        "--cpu=core-avx2",
        "--enable-llvm-intrinsics",
        include_opt.c_str()};
    luisa::string ispc_opt_string{ispc_options.front()};
    for (auto o : std::span{ispc_options}.subspan(1u)) {
        ispc_opt_string.append(" ").append(o);
    }
    auto cache_dir_str = ctx.cache_directory().string();
    auto source_path = ctx.cache_directory() / "kernel.ispc";
    auto llvm_ir_path = ctx.cache_directory() / "kernel.bc";
    {// dump source
        std::ofstream src_file{source_path};
        src_file << code;
    }
    auto compile_command = fmt::format(
        R"({} {} "{}" --emit-llvm -o "{}")",
        ispc_exe, ispc_opt_string, source_path.string(), llvm_ir_path.string());
    LUISA_INFO("Compiling ISPC kernel: {}", compile_command);
    if (auto ret = system(compile_command.c_str()); ret != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to compile ISPC kernel. "
            "Return code: {}.",
            ret);
    }
    return JITModule::load(llvm_ir_path);
}
//#else
//#error Unsupported platform for ISPC backend.
//#endif

}// namespace lc::ispc
