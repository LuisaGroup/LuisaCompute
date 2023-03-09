target("lc-core")
if _config_project ~= nil then
	_config_project({
		project_kind = "shared",
		batch_size = 4
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
-- if is_plat("windows") then
-- 	add_defines("_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS", "_CRT_SECURE_NO_WARNINGS",
-- 					"_ENABLE_EXTENDED_ALIGNED_STORAGE", {
-- 						public = true
-- 					})
-- end
if not EnableDSL then
	add_defines("LC_DISABLE_DSL", {
		public = true
	})
end
add_defines("LC_CORE_EXPORT_DLL")
