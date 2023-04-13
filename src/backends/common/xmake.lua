target("lc-backend-common")
_config_project({
	project_kind = "static",
	batch_size = 4
})
add_deps("lc-core")
add_files("string_scratch.cpp", "default_binary_io.cpp")
if LCCudaBackend and LCVulkanPath then
	set_values("vk_path", LCVulkanPath)
	set_values("vulkan_macro", "LUISA_CUDA_ENABLE_VULKAN_SWAPCHAIN")
	set_values("enable_swapchain", true)
	add_rules("lc_vulkan")
	add_files("vulkan_swapchain.cpp")
end
if LCCpuBackend then
	add_deps("lc-rust")
	add_files("rust_device_common.cpp")
end
