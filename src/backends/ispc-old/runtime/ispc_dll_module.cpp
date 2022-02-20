#pragma vengine_package ispc_vsproject
//
// Created by Mike on 2021/11/16.
//

#include <backends/ispc/runtime/ispc_dll_module.h>

namespace lc::ispc {

luisa::unique_ptr<Module> DLLModule::load(
    const Context &ctx, const std::filesystem::path &obj_path) noexcept {

    auto support_dir = ctx.runtime_directory() / "backends" / "ispc_support";
    auto link_exe = support_dir / "link.exe";
    auto crt_path = support_dir / "msvcrt.lib";
    auto dll_path = obj_path;
    auto lib_path = obj_path;
    auto exp_path = obj_path;
    dll_path.replace_extension("dll");
    lib_path.replace_extension("lib");
    exp_path.replace_extension("exp");

    auto command = fmt::format(
        R"({} /DLL /NOLOGO /OUT:"{}" /DYNAMICBASE "{}" /DEBUG:NONE /NOENTRY /EXPORT:run /NODEFAULTLIB "{}")",
        link_exe.string(),
        dll_path.string(),
        crt_path.string(),
        obj_path.string());
    LUISA_INFO("Generating DLL for ISPC kernel: {}", command);
    if (auto exit_code = system(command.c_str()); exit_code != 0) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to generate DLL for IPSC kernel. "
            "Linker exit with 0x{:02x}.",
            exit_code);
    }
    std::filesystem::remove(lib_path);
    std::filesystem::remove(exp_path);
    return luisa::make_unique<DLLModule>(DLLModule{DynamicModule{dll_path.parent_path(), dll_path.stem().string()}});
}

}// namespace lc::ispc
