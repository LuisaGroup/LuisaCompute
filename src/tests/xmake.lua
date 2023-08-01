local enable_gui = get_config("enable_gui")
target("stb-image")
	set_basename("lc-ext-stb-image")
	_config_project({
		project_kind = "shared"
	})
	add_headerfiles("../ext/stb/**.h")
	add_files("../ext/stb/stb.c")
	add_includedirs("../ext/stb", {
		public = true
	})
target_end()

-- TEST MAIN with doctest
------------------------------------
local function lc_add_app(appname, folder, name, options) 
	target(appname)
	_config_project({
		project_kind = "binary"
	})
	add_files("common/test_main.cpp")
	add_files("common/test_math_util.cpp")
	add_includedirs("./", {
		public = true
	})

	local match_str = path.join(name, "**.cpp")
	if name == "all" then
		match_str = "**.cpp"
	end
	set_pcxxheader("pch.h")
	add_files(path.join("next", folder, match_str))
	add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image", "lc-backends-dummy")
	if get_config("enable_ir") then
		add_deps("lc-ir")
		add_deps("lc-rust")
	end
	if get_config("enable_gui") then
		add_deps("lc-gui")
	end
	target_end()
end 

-- temp test suites
lc_add_app("test_feat", "test", "feat")

-- for common features

if get_config("enable_gui") then
	add_defines("ENABLE_DISPLAY")
	-- all test suites for release
	lc_add_app("test_all", "test", "all")
	-- example app 
	lc_add_app("gallary", "example", "gallary")
end
-- lc_add_app("test_io", "test", "io")
------------------------------------
-- TEST MAIN end

-- OLD TESTS

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
	if get_config("enable_ir") then
		add_deps("lc-ir")
		add_deps("lc-rust")
	end
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
if get_config("enable_ir") then
	test_proj('test_autodiff')
end
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
test_proj("test_atomic_queue", true)
test_proj("test_shared_memory", true)
test_proj("test_native_include", true)
test_proj("test_sparse_texture", true)
test_proj("test_dml")

if get_config("cuda_ext_lcub") then 
	test_proj("test_cuda_lcub", false, function ()
		add_deps("lc-backend-cuda-ext-lcub")
	end)
end

local enable_fsr2
local enable_xess
-- Super-sampling example
-- For FSR2, you need to clone https://github.com/GPUOpen-Effects/FidelityFX-FSR2 into this directory and compile
-- For XeSS, you need to clone https://github.com/intel/xess release package into this directory
-- enable_fsr2 = true
-- enable_xess = true
if get_config("dx_backend") and (enable_fsr2 or enable_xess) then
	test_proj("test_dx_supersampling", true, function()
		if enable_fsr2 then
			set_values("option", 1)
		else
			set_values("option", 2)
		end
		on_load(function(target)
			local function rela(p)
				return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
			end
			local option = target:values("option")
			if option == 1 then
				target:add("linkdirs", rela("FidelityFX-FSR2/bin/ffx_fsr2_api"))
				target:add("syslinks", "Advapi32", "User32")
				if is_mode("debug") then
					target:add("links", "ffx_fsr2_api_dx12_x64d", "ffx_fsr2_api_x64d")
				else
					target:add("links", "ffx_fsr2_api_dx12_x64", "ffx_fsr2_api_x64")
				end
				target:add("includedirs", rela("FidelityFX-FSR2/src/ffx-fsr2-api"))
				target:add("defines", "ENABLE_FSR")
			elseif option == 2 then
				target:add("links", rela("xess/lib/libxess"))
				target:add("includedirs", rela("xess/inc"))
			end
		end)
		after_build(function(target)
			local bin_dir = target:targetdir()
			local option = target:values("option")
			if option == 1 then
				local src_dir = path.join(os.scriptdir(), "FidelityFX-FSR2/bin")
				if is_mode("debug") then
					os.cp(path.join(src_dir, "ffx_fsr2_api_dx12_x64d.dll"), bin_dir)
					os.cp(path.join(src_dir, "ffx_fsr2_api_x64d.dll"), bin_dir)
				else
					os.cp(path.join(src_dir, "ffx_fsr2_api_dx12_x64.dll"), bin_dir)
					os.cp(path.join(src_dir, "ffx_fsr2_api_x64.dll"), bin_dir)
				end
			else
				local src_dir = path.join(os.scriptdir(), "xess/bin")
				os.cp(path.join(src_dir, "*.dll"), bin_dir)
			end
		end)
	end)
end