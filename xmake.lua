set_xmakever("2.7.8")
add_rules("mode.release", "mode.debug")
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
-- enable tools module
option("enable_tools")
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
-- python runtime path
option("py_path")
set_default("")
set_showmenu(true)
option_end()
-- python version, depends on python runtime, for instance --py_version=3.10
option("py_version")
set_default("")
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
-- enable Unity3D plugin module
option("enable_unity3d_plugin")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- pre-defined options end
if is_arch("x64", "x86_64", "arm64") then
	LCUseMimalloc = get_config("enable_mimalloc")
	LCUseSIMD = get_config("enable_simd")
	-- test require dsl
	LCEnableTest = get_config("enable_tests")
	LCEnableDSL = get_config("enable_dsl") or LCEnableTest
	LCDxBackend = get_config("dx_backend") and is_plat("windows")
	-- TODO: require environment check
	LCCudaBackend = get_config("cuda_backend") and (is_plat("windows") or is_plat("linux"))
	LCMetalBackend = get_config("metal_backend") and is_plat("macosx")
	LCCpuBackend = get_config("cpu_backend")
	LCRemoteBackend = get_config("remote_backend")
	LCEnableIR = get_config("enable_ir") or LCMetalBackend or LCCpuBackend or LCRemoteBackend
	LCEnableAPI = get_config("enable_api")
	-- TODO: rust condition
	LCEnableRust = LCEnableIR or LCEnableAPI
	local py_version = get_config("py_version")
	local py_path = get_config("py_path")
	LCEnablePython = type(py_path) == "string" and string.len(py_path) > 0 and type(py_version) == "string" and
					                 string.len(py_version) > 0
	LCEnableGUI = get_config("enable_gui") or LCEnableTest or LCEnablePython
	if is_mode("debug") then
		set_targetdir("bin/debug")
	else
		set_targetdir("bin/release")
	end
	includes("xmake_func.lua")
	includes("src")
else
	target("_lc_illegal_env")
	set_kind("phony")
	on_load(function(target)
		utils.error("Illegal environment. Please check your compiler, architecture or platform.")
	end)
	target_end()
end
