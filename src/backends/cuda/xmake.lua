target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
if get_config("_lc_vk_path") then
	add_defines("LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
	add_rules("lc_vulkan")
	add_deps("lc-vk-swapchain")
end
add_deps("lc-runtime", "lc-backend-common")
add_files("**.cpp")
on_load(function(target)
	local cuda_path = os.getenv("CUDA_PATH")
	if cuda_path then
		target:add("includedirs", path.join(cuda_path, "include/"))
		if is_plat("windows") then
			target:add("linkdirs", path.join(cuda_path, "lib/x64/"))
		elseif is_plat("linux") then
			target:add("linkdirs", path.join(cuda_path, "lib64/"))
		end
		target:add("links", "nvrtc", "cudart", "cuda")
	else
		target:set("enabled", false)
		return
	end
	if is_plat("windows") then
		target:add("defines", "UNICODE")
		target:add("syslinks", "Cfgmgr32", "Advapi32")
	end
end)
target_end()
