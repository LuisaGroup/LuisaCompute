set_xmakever("2.7.8")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
-- disable ccache in-case error
set_policy("build.ccache", false)
-- pre-defined options
-- enable mimalloc as default allocator: https://github.com/LuisaGroup/mimalloc
option("enable_mimalloc")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable unity(jumbo) build, enable this option will optimize compile speed
option("enable_unity_build")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable sse and sse2 SIMD
option("enable_simd")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable DirectX-12 backend
option("dx_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable Vulkan backend
option("vk_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable NVIDIA-CUDA backend
option("cuda_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable Metal backend
option("metal_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()
-- enable cpu backend
option("cpu_backend")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable tests module
option("enable_tests")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("py_include")
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("py_linkdir")
set_default(false)
set_showmenu(true)
option_end()
-- python include path
option("py_libs")
set_default(false)
set_showmenu(true)
option_end()
-- enable intermediate representation module (rust required)
option("enable_ir")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable c-language api module for cross-language bindings module
option("enable_api")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable C++ DSL module
option("enable_dsl")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- enable GUI module
option("enable_gui")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- custom bin dir
option("bin_dir")
set_default("bin")
set_showmenu(true)
option_end()
-- pre-defined options end
if is_arch("x64", "x86_64", "arm64") then
	option("_lc_vk_path")
	set_default(false)
	set_showmenu(false)
	after_check(function(option)
		local path = os.getenv("VULKAN_SDK")
		if not path then
			path = os.getenv("VK_SDK_PATH")
		end
		option:set_value(path)
	end)
	option_end()
	option("_lc_inter_bin_dir")
	set_default(false)
	set_showmenu(false)
	add_deps("bin_dir")
	after_check(function(option)
		if path.absolute(os.projectdir()) == path.absolute(os.scriptdir()) then
			local bin_dir = option:dep("bin_dir"):enabled()
			if is_mode("debug") then
				bin_dir = path.join(bin_dir, "debug")
			elseif is_mode("releasedbg") then
				bin_dir = path.join(bin_dir, "releasedbg")
			else
				bin_dir = path.join(bin_dir, "release")
			end
			option:set_value(bin_dir)
		else
			option:set_value(false)
		end
	end)
	option_end()
	LCUseMimalloc = get_config("enable_mimalloc")
	LCUseSIMD = get_config("enable_simd")
	-- test require dsl
	LCEnableTest = get_config("enable_tests")
	LCEnableDSL = get_config("enable_dsl") or LCEnableTest
	LCDxBackend = get_config("dx_backend") and is_plat("windows")

	local bin_dir = get_config("_lc_inter_bin_dir")
	if (bin_dir) then
		set_targetdir(bin_dir)
	end
	LCVulkanPath = get_config("_lc_vk_path")
	LCVkBackend = get_config("vk_backend") and LCVulkanPath
	-- TODO: require environment check
	LCCudaBackend = get_config("cuda_backend") and (is_plat("windows") or is_plat("linux"))
	LCMetalBackend = get_config("metal_backend") and is_plat("macosx")
	LCCpuBackend = get_config("cpu_backend")
	LCRemoteBackend = get_config("remote_backend")
	LCEnableIR = get_config("enable_ir") or LCMetalBackend or LCCpuBackend or LCRemoteBackend
	LCEnableAPI = get_config("enable_api")
	-- TODO: rust condition
	LCEnableRust = LCEnableIR or LCEnableAPI

	LCEnablePython = type(get_config("py_include")) == "string"
	LCEnableGUI = get_config("enable_gui") or LCEnableTest or LCEnablePython
	includes("xmake_lua/xmake_func.lua")
	includes("src")
else
	target("_lc_illegal_env")
	set_kind("phony")
	on_load(function(target)
		utils.error("Illegal environment. Please check your compiler, architecture or platform.")
	end)
	target_end()
end
