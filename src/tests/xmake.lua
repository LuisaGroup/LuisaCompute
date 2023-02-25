_config_project({
	project_name = "lc_test",
	project_kind = "binary"
})
add_includedirs("../ext/stb")
-- add_files("test_dsl.cpp")
-- add_files("test_dynamic_buffer.cpp")
add_files("test_path_tracing.cpp")
-- add_files("test_texture_compress.cpp", "../ext/stb/stb.c")
-- add_files("test_raster.cpp", "../ext/stb/stb.c")
-- add_files("test_dispatch_indirect.cpp")
add_deps("lc-runtime", "lc-vstl", "lc-gui")
if EnableDSL then
	add_deps("lc-dsl")
end
