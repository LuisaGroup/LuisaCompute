target("lc-core")
set_basename("luisa-core")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
on_load(function(target)
    local function rela(p)
        return path.normalize(path.join(os.scriptdir(), p))
    end
    target:add("includedirs", rela("../../include"), rela("../ext/xxHash/"), rela("../ext/magic_enum/include"),
        rela("../ext/half/include"), {
            public = true
        })
    if target:is_plat("windows") then
        if is_mode("debug") then
            target:add("syslinks", "Dbghelp")
            if has_config('lc_disable_win_message_box') and target:is_plat('windows') then
                target:add('defines', 'LUISA_DISABLE_WIN_MESSAGE_BOX')
            end
        end
        target:add("defines", "NOMINMAX", "_CRT_SECURE_NO_WARNINGS", "LUISA_PLATFORM_WINDOWS", "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR", {
            public = true
        })
    elseif target:is_plat("linux") then
        target:add("defines", "LUISA_PLATFORM_UNIX", {
            public = true
        })
    elseif target:is_plat("macosx") then
        target:add("defines", "LUISA_PLATFORM_UNIX", "LUISA_PLATFORM_APPLE", {
            public = true
        })
    end
    if has_config("lc_enable_dsl") then
        target:add("defines", "LUISA_ENABLE_DSL", {
            public = true
        })
    end
    if has_config("lc_use_system_stl") then
        target:add("defines", "LUISA_USE_SYSTEM_STL", "_ENABLE_EXTENDED_ALIGNED_STORAGE", {
            public = true
        })
    end
    target:add("defines", "LUISA_CORE_EXPORT_DLL")
    target:add("deps", "eastl")
    if has_config("lc_spdlog_use_xrepo") then
        target:add("packages", "spdlog")
    else
        target:add("deps", "spdlog")
    end
    target:add("deps", "lc-check-winsdk")
    if has_config("spdlog_only_fmt") then -- Use no spdlog
        target:add("defines", "LUISA_CUSTOM_LOGGER", {
            public = true
        })
    end
    local marl_path = path.join(os.scriptdir(), "../ext/marl")
    if (not has_config("lc_external_marl")) and (os.exists(marl_path)) then
        target:add("defines", "MARL_DLL", {
            public = true
        })
        target:add("defines", "MARL_BUILDING_DLL")
        target:add("files", path.join(marl_path, "src/build.marl.cpp"))
        -- Only include architecture-specific osfiber sources for the target arch.
        -- Globbing all *.c and *.S causes cross-arch compile errors (e.g. -mfpu= on ARM64 macOS).
        if not target:is_plat("windows") then
            local arch = target:arch()
            if arch == "arm64" or arch == "aarch64" or arch == "arm64-v8a" then
                target:add("files", path.join(marl_path, "src/osfiber_aarch64.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_aarch64.S"))
            elseif arch == "x86_64" or arch == "x64" then
                target:add("files", path.join(marl_path, "src/osfiber_x64.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_x64.S"))
            elseif arch == "x86" or arch == "i386" then
                target:add("files", path.join(marl_path, "src/osfiber_x86.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_x86.S"))
            elseif arch == "arm" or arch == "armv7" or arch == "armeabi-v7a" then
                target:add("files", path.join(marl_path, "src/osfiber_arm.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_arm.S"))
            elseif arch == "mips64" or arch == "mips64el" then
                target:add("files", path.join(marl_path, "src/osfiber_mips64.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_mips64.S"))
            elseif arch == "loongarch64" then
                target:add("files", path.join(marl_path, "src/osfiber_loongarch64.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_loongarch64.S"))
            elseif arch == "riscv64" or arch == "rv64" then
                target:add("files", path.join(marl_path, "src/osfiber_rv64.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_rv64.S"))
            elseif arch == "ppc64" or arch == "ppc64le" then
                target:add("files", path.join(marl_path, "src/osfiber_ppc64.c"))
                target:add("files", path.join(marl_path, "src/osfiber_asm_ppc64.S"))
            end
        end
        target:add("includedirs", path.join(marl_path, "include"), {
            public = true
        })
    end
end)
add_headerfiles("../../include/luisa/core/**.h", "../ext/xxHash/**.h", "../ext/magic_enum/include/**.hpp",
    "../ext/half/include/half.hpp") -- , "../ext/parallel-hashmap/**.h"
add_files("**.cpp")
target_end()
