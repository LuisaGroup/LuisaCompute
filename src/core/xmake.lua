target("lc-core")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
if is_plat("windows") then
    if is_mode("debug") then
        add_syslinks("Dbghelp")
    end
	add_defines("NOMINMAX", "LUISA_PLATFORM_WINDOWS", "LUISA_USE_DIRECT_STORAGE", {public = true})
elseif is_plat("linux") then
    add_defines("LUISA_PLATFORM_UNIX", {public = true})
elseif is_plat("macosx") then
    add_defines("LUISA_PLATFORM_UNIX", "LUISA_PLATFORM_APPLE", {public = true})
end
add_deps("eastl", "spdlog")
add_includedirs("../", "../ext/xxHash/", "../ext/magic_enum/include", "../ext/parallel-hashmap", {
    public = true
})
add_files("**.cpp")
-- if is_plat("windows") then
-- 	add_defines("_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS", "_CRT_SECURE_NO_WARNINGS",
-- 					"_ENABLE_EXTENDED_ALIGNED_STORAGE", {
-- 						public = true
-- 					})
-- end
if get_config("enable_dsl") then
    add_defines("LUISA_ENABLE_DSL", {
        public = true
    })
end
add_defines("LC_CORE_EXPORT_DLL")
target_end()
