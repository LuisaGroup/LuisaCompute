target("lc-backend")
_config_project({
	project_kind = "static"
})
add_files("**.cpp")
add_deps("lc-runtime")
if DxBackend then
	add_defines("LC_ENABLE_DX_BACKEND")
end
if CudaBackend then
	add_defines("LC_ENABLE_CUDA_BACKEND")
end
if MetalBackend then
	add_defines("LC_ENABLE_METAL_BACKEND")
end
if CpuBackend then
	add_defines("LC_ENABLE_CPU_BACKEND")
end