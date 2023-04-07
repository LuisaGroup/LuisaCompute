target("lc-core")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
if is_mode("debug") and is_plat("windows") then
	add_syslinks("Dbghelp")
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
if LCEnableDSL then
	add_defines("LUISA_ENABLE_DSL", {
		public = true
	})
end
add_defines("LC_CORE_EXPORT_DLL")
target_end()
