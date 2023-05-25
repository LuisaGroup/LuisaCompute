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
local function test_proj(name, gui_dep, callable)
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
	if callable then
		callable()
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
test_proj("test_denoiser", true)
test_proj("test_sdf_renderer", true, function()
	add_defines("ENABLE_DISPLAY")
end)
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
test_proj("test_dstorage", true)
test_proj("test_indirect", true)
test_proj("test_texture3d", true)
if get_config("dx_backend") then
	test_proj("test_dx_fsr2", true, function()
		add_linkdirs("FidelityFX-FSR2/bin/ffx_fsr2_api")
		add_syslinks("Advapi32", "User32")
		if is_mode("debug") then
			add_links("ffx_fsr2_api_dx12_x64d", "ffx_fsr2_api_x64d")
		else
			add_links("ffx_fsr2_api_dx12_x64", "ffx_fsr2_api_x64")
		end
		add_includedirs("FidelityFX-FSR2/src/ffx-fsr2-api")
		after_build(function(target)
			local bin_dir = target:targetdir()
			local src_dir = path.join(os.scriptdir(), "FidelityFX-FSR2/bin")
			if is_mode("debug") then
				os.cp(path.join(src_dir, "ffx_fsr2_api_dx12_x64d.dll"), bin_dir)
				os.cp(path.join(src_dir, "ffx_fsr2_api_x64d.dll"), bin_dir)
			else
				os.cp(path.join(src_dir, "ffx_fsr2_api_dx12_x64.dll"), bin_dir)
				os.cp(path.join(src_dir, "ffx_fsr2_api_x64.dll"), bin_dir)
			end
		end)
	end)
end
