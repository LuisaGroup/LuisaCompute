includes("build_proj.lua")
if UseMimalloc then
	_configs.enable_mimalloc = true
end
includes("ext/EASTL")
_configs.enable_mimalloc = nil
includes("ext/spdlog")
includes("core")
includes("vstl")
includes("ast")
includes("runtime")
if EnableDSL then
	includes("dsl")
end
if EnableGUI then
	includes("gui")
end
if EnablePython then
	includes("py")
end
includes("backends/common")
if DxBackend then
	includes("backends/dx")
end
if CudaBackend then
	includes("backends/cuda")
end
if MetalBackend then
	includes("backends/metal")
end
if CpuBackend then
	includes("backends/cpu")
end
if EnableTest then
	includes("tests")
end
if get_config("enable_tools") then
	includes("tools")
end
if EnableRust then
	includes("rust")
end
if EnableIR then
	includes("ir")
end
if EnableAPI then
	includes("api")
end
if get_config("enable_unity3d_plugin") then
	includes("unity3d")
end