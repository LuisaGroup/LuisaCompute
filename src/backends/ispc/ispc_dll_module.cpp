//
// Created by Mike on 2021/11/16.
//

#include <core/platform.h>
#include <core/logging.h>
#include <core/dynamic_module.h>
#include <runtime/context.h>
#include <backends/ispc/ispc_dll_module.h>

namespace luisa::compute::ispc {

luisa::shared_ptr<ISPCModule> ISPCDLLModule::load(
    const Context &ctx, const std::filesystem::path &obj_path) noexcept {

#ifdef LUISA_PLATFORM_WINDOWS
    auto &&support_dir = ctx.runtime_directory();
    auto link_exe = support_dir / "link.exe";
    auto crt_path = support_dir / "msvcrt.lib";
    auto embree_path = support_dir / "embree3.lib";
    auto dll_path = obj_path;
    auto lib_path = obj_path;
    auto exp_path = obj_path;
    dll_path.replace_extension("dll");
    lib_path.replace_extension("lib");
    exp_path.replace_extension("exp");

    auto command = luisa::format(
        R"({} /DLL /NOLOGO /OUT:"{}" /DYNAMICBASE "{}" "{}" /DEBUG:NONE /NOENTRY /EXPORT:kernel_main /NODEFAULTLIB "{}")",
        link_exe.string(), dll_path.string(), crt_path.string(), embree_path.string(), obj_path.string());
    LUISA_INFO("Generating DLL for ISPC kernel: {}", command);
    if (auto exit_code = system(command.c_str()); exit_code != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to generate DLL for IPSC kernel. "
            "Linker exit with 0x{:02x}.",
            exit_code);
    }
    std::filesystem::remove(lib_path);
    std::filesystem::remove(exp_path);
    return luisa::make_unique<ISPCDLLModule>(ISPCDLLModule{
        DynamicModule{dll_path.parent_path(), dll_path.stem().string()}});
#else

    auto embree_name = [&ctx] {
        std::filesystem::directory_iterator runtime_files{ctx.runtime_directory()};
        for (auto &&file : runtime_files) {
            auto filename = file.path().filename();
            auto stem = filename.stem().string();
            auto ext = filename.extension().string();
            if (stem.find("embree3") != std::string::npos &&
                (ext == ".dll" || ext == ".so" || ext == ".dylib")) {
                return stem.starts_with("lib") ? stem.substr(3u) : stem;
            }
        }
        LUISA_WARNING_WITH_LOCATION("Failed to find embree3.");
        return std::string{};
    }();

    auto support_dir = ctx.runtime_directory() / "backends" / "ispc_support";
    auto file_name = obj_path.filename().stem().string();
    auto output_folder = obj_path.parent_path();
    auto dll_path = output_folder / luisa::format("lib{}.so", file_name);

#ifndef NDEBUG
    auto link_opt = "-shared -g -O3 -flto";
#else
    auto link_opt = "-shared -O3 -flto";
#endif

    auto command = luisa::format(
        R"(cc {} -o "{}" "{}")",
        link_opt, dll_path.string(), obj_path.string());
    if (!embree_name.empty()) {
        command.append(luisa::format(
            R"( -L"{}" -l{})",
            ctx.runtime_directory().string(),
            embree_name));
    }
    LUISA_INFO("Generating DLL for ISPC kernel: {}", command);
    if (auto exit_code = system(command.c_str()); exit_code != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to generate DLL for IPSC kernel. "
            "Linker exit with 0x{:02x}.",
            exit_code);
    }
    return luisa::make_shared<ISPCDLLModule>(ISPCDLLModule{
        DynamicModule{output_folder, file_name}});
#endif
}

}// namespace luisa::compute::ispc
