target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
if LCVulkanPath then
	set_values("vk_path", LCVulkanPath)
	set_values("vulkan_macro", "LUISA_CUDA_ENABLE_VULKAN_SWAPCHAIN")
	set_values("enable_swapchain", true)
	add_rules("lc_vulkan")
end
add_deps("lc-runtime")
add_files("**.cpp", "../common/default_binary_io.cpp", "../common/string_scratch.cpp")
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
		target:add("defines", "NOMINMAX", "UNICODE")
		target:add("syslinks", "Cfgmgr32", "Advapi32")
	end
end)
-- add_includedirs("#")
after_build(function(target)
	local binDir = target:targetdir()
	os.cp(path.join(os.scriptdir(), "cuda_builtin"), path.join(binDir, ".data/"))
end)
target_end()
