target("lc-backend")
_config_project({
	project_kind = "static"
})
-- files
add_files("default_binary_io.cpp")
if MetalBackend or CudaBackend then
	add_files("string_scratch.cpp")
end
if CpuBackend or RemoteBackend then
	add_files("rust_device_common.cpp")
end
add_deps("lc-runtime")
