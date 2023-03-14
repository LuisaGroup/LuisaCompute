set_xmakever("2.7.3")
add_rules("mode.release", "mode.debug")
-- pre-defined options
option("enable_mimalloc")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("enable_unity_build")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("enable_simd")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("dx_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("cuda_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("metal_backend")
set_values(true, false)
set_default(true)
set_showmenu(true)
option_end()

option("cpu_backend")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("enable_tools")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("enable_tests")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("py_path")
set_default("")
set_showmenu(true)
option_end()

option("py_version")
set_default("")
set_showmenu(true)
option_end()

option("enable_ir")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("enable_api")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("enable_dsl")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("enable_gui")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()

option("enable_unity3d_plugin")
set_values(true, false)
set_default(false)
set_showmenu(true)
option_end()
-- pre-defined options end

if is_arch("x64", "x86_64", "arm64") then
	UseMimalloc = get_config("enable_mimalloc")
	UseSIMD = get_config("enable_simd")
	-- test require dsl
	EnableTest = get_config("enable_tests")
	EnableDSL = get_config("enable_dsl") or EnableTest
	DxBackend = get_config("dx_backend") and is_plat("windows")
	-- TODO: require environment check
	CudaBackend = get_config("cuda_backend") and (is_plat("windows") or is_plat("linux")) and false
	MetalBackend = get_config("metal_backend") and is_plat("macos")
	CpuBackend = get_config("cpu_backend")
	EnableIR = get_config("enable_ir") or CudaBackend or MetalBackend or CpuBackend
	EnableAPI = get_config("enable_api")
	-- TODO: rust condition
	EnableRust = EnableIR or EnableAPI
	PythonVersion = get_config("py_version")
	PythonPath = get_config("py_path")
	EnablePython = type(PythonPath) == "string" and type(PythonVersion) == "string" and PythonPath:len() > 0 and
					               PythonVersion:len() > 0
	EnableGUI = get_config("enable_gui") or EnableTest or EnablePython

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
end
