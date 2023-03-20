target("lc-backend")
_config_project({
	project_kind = "static"
})
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
-- files
add_files("default_binary_io.cpp")
if MetalBackend or CudaBackend then
	add_files("string_scratch.cpp")
end
if CpuBackend or RemoteBackend then
	add_files("rust_device_common.cpp")
end
add_deps("lc-runtime")
