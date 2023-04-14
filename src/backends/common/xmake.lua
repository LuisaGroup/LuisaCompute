target("lc-backend-common")
_config_project({
	project_kind = "static",
	batch_size = 4
})
add_deps("lc-core")
add_files("string_scratch.cpp", "default_binary_io.cpp")
if (LCCudaBackend or LCCpuBackend) and LCVulkanPath then
	set_values("vk_path", LCVulkanPath)
	set_values("vk_public", true)
	add_rules("lc_vulkan")
	add_files("vulkan_swapchain.cpp")
end
if LCCpuBackend then
	add_deps("lc-rust")
	add_files("rust_device_common.cpp")
end
