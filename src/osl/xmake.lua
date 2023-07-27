target("lc-osl")
_config_project({
	project_kind = "shared",
	batch_size = 16
})
add_defines("LC_OSL_EXPORT_DLL")
add_deps("lc-ast", "lc-runtime")
add_headerfiles("../../include/luisa/osl/**.h")
add_files("**.cpp")
