target("lc-core")
if _config_project ~= nil then
_config_project({
	project_kind = "shared",
	batch_size = 8
})	
end
if is_mode("debug") and is_plat("windows") then
    add_syslinks("Dbghelp")
end
add_deps("eastl", "spdlog")
add_includedirs("../", "../ext/xxHash/", "../ext/parallel-hashmap", {
    public = true
})
add_files("**.cpp")
add_defines("UNICODE=1", "NOMINMAX=1", "_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1",
        "_CRT_SECURE_NO_WARNINGS=1", "_ENABLE_EXTENDED_ALIGNED_STORAGE=1", {
            public = true
        })
if not EnableDSL then
    add_defines("LC_DISABLE_DSL", {
        public = true
    })
end
add_defines("LC_CORE_EXPORT_DLL")
