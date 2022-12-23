_config_project({
	project_name = "lc-runtime",
	project_kind = "shared",
	batch_size = 4
})
add_deps("lc-ast")
add_defines("LC_RUNTIME_EXPORT_DLL")
add_files("**.cpp", "../raster/**.cpp", "../rtx/**.cpp")
