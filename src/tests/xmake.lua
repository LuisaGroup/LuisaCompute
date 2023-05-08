local enable_gui = get_config("enable_gui")
target("stb-image")
set_basename("lc-ext-stb-image")
_config_project({
	project_kind = "shared"
})
add_files("../ext/stb/stb.c")
add_includedirs("../ext/stb", {
	public = true
})
target_end()
local function test_proj(name, gui_dep, defines)
	if gui_dep and not enable_gui then
		return
	end
	target(name)
	_config_project({
		project_kind = "binary"
	})
	add_files(name .. ".cpp")
	add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image", "lc-backends-dummy")
	if get_config("enable_gui") then
		add_deps("lc-gui")
	end
	if defines then
		for i, v in ipairs(defines) do
			add_defines(v)
		end
	end
	target_end()
end

-- FIXME: @Maxwell please use the doctest framework
test_proj("test_helloworld")
test_proj("test_ast")
test_proj("test_atomic")
test_proj("test_bindless", true)
test_proj("test_callable")
-- test_proj("test_dsl")
test_proj("test_dsl_multithread")
test_proj("test_dsl_sugar")
test_proj("test_game_of_life", true)
test_proj("test_mpm3d", true)
test_proj("test_mpm88", true)
test_proj("test_normal_encoding")
test_proj("test_path_tracing", true)
test_proj("test_path_tracing_camera", true)
test_proj("test_path_tracing_cutout", true)
test_proj("test_photon_mapping", true)
test_proj("test_printer")
test_proj("test_procedural")
test_proj("test_rtx")
test_proj("test_runtime", true)
test_proj("test_sampler")
test_proj("test_denoiser",true)
test_proj("test_sdf_renderer", true, {"ENABLE_DISPLAY"})
test_proj("test_shader_toy", true)
test_proj("test_shader_visuals_present", true)
test_proj("test_texture_io")
test_proj("test_thread_pool")
test_proj("test_type")
test_proj("test_raster", true)
test_proj("test_texture_compress")
test_proj("test_swapchain", true)
test_proj("test_swapchain_static", true)
test_proj("test_select_device", true)
