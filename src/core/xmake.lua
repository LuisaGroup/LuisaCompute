target("lc-core")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
on_load(function(target)
    local function rela(p)
        return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
    end
    target:add("includedirs", rela("../../include"), rela("../ext/xxHash/"), rela("../ext/magic_enum/include"),
        rela("../ext/half/include"), {
            public = true
        })
    if is_plat("windows") then
        if is_mode("debug") then
            target:add("syslinks", "Dbghelp")
        end
        target:add("defines", "NOMINMAX", "LUISA_PLATFORM_WINDOWS", {
            public = true
        })
    elseif is_plat("linux") then
        target:add("defines", "LUISA_PLATFORM_UNIX", {
            public = true
        })
    elseif is_plat("macosx") then
        target:add("defines", "LUISA_PLATFORM_UNIX", "LUISA_PLATFORM_APPLE", {
            public = true
        })
    end
    if get_config("enable_dsl") then
        target:add("defines", "LUISA_ENABLE_DSL", {
            public = true
        })
    end
    target:add("defines", "LC_CORE_EXPORT_DLL")
    if is_plat("windows") then
        target:add("defines", "_CRT_SECURE_NO_WARNINGS")
    end
    target:add("deps", "eastl", "spdlog", "lc-check-winsdk")
    local marl_path = path.join(os.projectdir(), "src/ext/marl")
    if os.exists(marl_path) then
        target:add("defines", "MARL_DLL", {public = true})
        target:add("defines", "MARL_BUILDING_DLL")
        target:add("files", path.join(marl_path, "src/*.c"))
        if not is_plat("windows") then
            target:add("files", path.join(marl_path, "src/*.S"))
        end
        target:add("includedirs", path.join(marl_path, "include"), {
            public = true
        })
        local src_path = path.join(marl_path, "src")
        for _, filepath in ipairs(os.files(path.join(src_path, "*.cpp"))) do
            local file_name = path.filename(filepath)
            if not (file_name:find("_bench") or file_name:find("_test")) then
                target:add("files", filepath)
            end
        end
    end
end)
add_headerfiles("../../include/luisa/core/**.h", "../ext/xxHash/**.h", "../ext/magic_enum/include/**.hpp",
    "../ext/half/include/half.hpp") -- , "../ext/parallel-hashmap/**.h"
add_files("**.cpp")
target_end()
