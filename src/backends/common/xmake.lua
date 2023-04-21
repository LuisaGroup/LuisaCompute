target("lc-backend-common")
_config_project({
	project_kind = "static",
	batch_size = 4
})
add_deps("lc-core")
add_files("string_scratch.cpp", "default_binary_io.cpp")
if get_config("cpu_backend") then
	add_deps("lc-rust")
	add_files("rust_device_common.cpp")
end
if get_config("vk_backend") or get_config("dx_backend") then
	add_files("hlsl/*.cpp")
	includes("hlsl/builtin")
end

if (get_config("cuda_backend") or get_config("cpu_backend")) and get_config("_lc_vk_path") then
	target("lc-backend-vk-swapchain")
	_config_project({
		project_kind = "shared"
	})
	set_values("vk_public", true)
	add_rules("lc_vulkan")
	add_files("vulkan_swapchain.cpp")
	add_deps("lc-core")
	add_defines("LC_VK_SWAPCHAIN_EXPORT")
	target_end()
end
