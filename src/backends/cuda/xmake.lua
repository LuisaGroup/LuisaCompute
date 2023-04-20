target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
if LCVulkanPath then
	add_defines("LUISA_CUDA_ENABLE_VULKAN_SWAPCHAIN")
	set_values("vk_path", LCVulkanPath)
	add_rules("lc_vulkan")
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
-- add_includedirs("#")
after_build(function(target)
	local bin_dir = target:targetdir()
	local data_dir = path.join(bin_dir, ".data/cuda_builtin")
	os.mkdir(data_dir)
	os.cp(path.join(os.scriptdir(), "cuda_builtin/*"), data_dir)
end)
target_end()
