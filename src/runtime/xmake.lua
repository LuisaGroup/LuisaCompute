_config_project({
	project_name = "lc-runtime",
	project_kind = "shared",
	batch_size = 4
})
add_deps("lc-ast")
add_defines("LC_RUNTIME_EXPORT_DLL")
if EnableRust then
	local add_defines = _get_add_defines()
	add_defines("LC_ENABLE_API", {
		public = true
	})
end
add_files("**.cpp", "../raster/**.cpp", "../rtx/**.cpp")
