target("stb-image")
_config_project({
	project_kind = "static"
})
add_files("../ext/stb/stb.c")
add_includedirs("../ext/stb", {
	public = true
})
target_end()
local function test_proj(name)
	target(name)
	_config_project({
		project_kind = "binary"
	})
	add_files(name .. ".cpp")
	add_deps("lc-runtime", "lc-vstl", "lc-gui", "stb-image")
	if EnableDSL then
		add_deps("lc-dsl")
	end
	target_end()
end
test_proj("test_ast")
test_proj("test_atomic")
test_proj("test_bindless")
test_proj("test_dsl")
test_proj("test_dsl_multithread")
test_proj("test_dsl_sugar")
test_proj("test_game_of_life")
test_proj("test_mpm3d")
test_proj("test_mpm88")
test_proj("test_normal_encoding")
test_proj("test_path_tracing")
test_proj("test_photon_mapping")
test_proj("test_printer")
test_proj("test_procedural")
test_proj("test_rtx")
test_proj("test_runtime")
test_proj("test_sampler")
test_proj("test_sdf_renderer")
add_defines("ENABLE_DISPLAY")
test_proj("test_shader_toy")
test_proj("test_shader_visuals_present")
test_proj("test_simple")
test_proj("test_texture_io")
test_proj("test_thread_pool")
test_proj("test_type")
