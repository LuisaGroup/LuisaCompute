includes("build_proj.lua")
if UseMimalloc then
	_configs.use_mimalloc = true
end
includes("ext/EASTL")
_configs.use_mimalloc = nil
includes("ext/spdlog")
-- includes("ext/glfw")
includes("core")
includes("vstl")
-- includes("meta")
includes("ast")
includes("runtime")
includes("compile")
if not DisableDSL then
	includes("dsl")	
end
includes("serde_lib")
if get_config("enable_py") then
	includes("py")
end
-- includes("remote")
if get_config("dx_backend") then
	includes("backends/dx")
end
if get_config("cuda_backend") then
	includes("backends/cuda")
end
if get_config("llvm_backend") then
	includes("backends/llvm")
end
if get_config("metal_backend") then
	includes("backends/metal")
end
if get_config("enable_tests") then
	includes("tests")
end
-- if is_plat("windows") then
-- includes("shader_graph")
-- end
if get_config("enable_tools") then
	includes("tools")
end