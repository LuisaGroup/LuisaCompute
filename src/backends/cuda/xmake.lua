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
add_deps("lc-runtime")
add_files("**.cpp","../common/default_binary_io.cpp", "../common/string_scratch.cpp")
on_load(function(target)
	import("detect.sdks.find_cuda")
	local cuda = find_cuda()
	if cuda then
		local linkdirs = cuda["linkdirs"]
		if linkdirs then
			for i,v in ipairs(linkdirs) do
				target:add("linkdirs", v)
			end
		end
		local includedirs = cuda["includedirs"]
		if includedirs then
			for i,v in ipairs(includedirs) do
				target:add("includedirs", v)
			end
		end
		target:add("links", "nvrtc", "cudart")
		target:add("syslinks", "cuda")
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
