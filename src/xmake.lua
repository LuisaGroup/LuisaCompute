includes("build_proj.lua")
if LCUseMimalloc then
	_configs.enable_mimalloc = true
end
includes("ext/EASTL")
_configs.enable_mimalloc = nil
includes("ext/spdlog")
includes("core")
includes("vstl")
includes("ast")
includes("runtime")
if LCEnableDSL then
	includes("dsl")
end
if LCEnableGUI then
	includes("gui")
end
if LCEnablePython then
	includes("py")
end
includes("backends/validation")
if LCDxBackend then
	includes("backends/dx")
end
if LCCudaBackend then
	includes("backends/cuda")
end
if LCMetalBackend then
	includes("backends/metal")
end
if LCCpuBackend then
	includes("backends/cpu")
end
if LCRemoteBackend then
	includes("backends/remote")
end
if LCEnableTest then
	includes("tests")
end
if get_config("enable_tools") then
	includes("tools")
end
if LCEnableRust then
	includes("rust")
end
if LCEnableIR then
	includes("ir")
end
if LCEnableAPI then
	includes("api")
end
if get_config("enable_unity3d_plugin") then
	includes("unity3d")
end
target("magic_enum")
	set_kind("headeronly")
	add_includedirs("src/ext/magic_enum/include")
target_end()