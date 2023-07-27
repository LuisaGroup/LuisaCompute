if get_config("vk_backend") or get_config("dx_backend") then
	includes("hlsl/builtin")
end

if (get_config("cuda_backend") or get_config("cpu_backend")) and get_config("_lc_vk_path") then
	target("lc-vulkan-swapchain")
	_config_project({
		project_kind = "shared"
	})
	set_values("vk_public", true)
	add_rules("lc_vulkan")
	add_headerfiles("vulkan_swapchain.h", "vulkan_instance.h")
	add_files("vulkan_swapchain.cpp", "vulkan_instance.cpp")
	add_deps("lc-core")
	if is_plat("linux") then
		add_syslinks("xcb", "X11")
	end
	add_defines("LC_BACKEND_EXPORT_DLL")
	target_end()
end
