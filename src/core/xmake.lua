_config_project({
    project_name = "lc-core",
    project_kind = "shared"
})
local add_includedirs = _get_add_includedirs()
local add_defines = _get_add_defines()
if is_mode("debug") and is_plat("windows") then
    add_syslinks("Dbghelp")
end
add_deps("eastl", "spdlog")
add_includedirs("../", "../ext/xxHash/", "../ext/parallel-hashmap", {
    public = true
})
add_files("**.cpp")
add_defines("UNICODE=1", "TSL_NO_EXCEPTIONS", "NOMINMAX=1", "_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1",
        "_CRT_SECURE_NO_WARNINGS=1", "_ENABLE_EXTENDED_ALIGNED_STORAGE=1", {
            public = true
        })
if not EnableDSL then
    add_defines("LC_DISABLE_DSL", {
        public = true
    })
end
add_defines("LC_CORE_EXPORT_DLL")
