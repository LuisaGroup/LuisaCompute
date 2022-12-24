_config_project({
	project_name = "lc-compile",
	project_kind = "shared"
})
add_deps("lc-ast")
add_defines("LC_COMPILE_EXPORT_DLL")
add_files("**.cpp")
