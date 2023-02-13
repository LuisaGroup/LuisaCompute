includes("build_proj.lua")
if UseMimalloc then
	_configs.use_mimalloc = true
end
includes("ext/EASTL")
_configs.use_mimalloc = nil
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
if get_config("enable_py") then
	includes("py")
end
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
	includes("api")
	includes("ir")
	includes("rust")
end