if get_config("cuda_ext_lcub") then 
	includes("lcub")
end

target("lc-backend-cuda")
_config_project({
	project_kind = "shared",
	batch_size = 4
})
if get_config("_lc_vk_path") then
	add_defines("LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
	add_rules("lc_vulkan")
	add_deps("lc-vulkan-swapchain")
end
add_deps("lc-runtime")
if get_config("enable_ir") then
	add_deps("lc-ir")
end
set_pcxxheader("pch.h")
add_headerfiles("*.h", "./cuda_builtin/*.h", "../common/default_binary_io.h", "../common/string_scratch.h")
add_files("*.cpp")

-- if has_config("cuda_ext_lcub") then
-- 	add_files("lcub/*.cpp")
-- 	add_deps("cuda-dcub")
-- end

on_load(function(target)
	import("detect.sdks.find_cuda")
	local cuda = find_cuda()
	if cuda then
		local function set(key, value)
			if type(value) == "string" then
				target:add(key, value)
			elseif type(value) == "table" then
				for i, v in ipairs(value) do
					target:add(key, v)
				end
			end
		end
		set("linkdirs", cuda["linkdirs"])
		set("includedirs", cuda["includedirs"])
		target:add("links", "nvrtc", "cuda")
	else
		target:set("enabled", false)
		return
	end
	if is_plat("windows") then
		target:add("defines", "UNICODE", "_CRT_SECURE_NO_WARNINGS")
		target:add("syslinks", "Cfgmgr32", "Advapi32")
	end
end)
after_build(function(target)
	import("detect.sdks.find_cuda")
	local cuda = find_cuda()
	if cuda then
		local linkdirs = cuda["linkdirs"]
		local bin_dir = target:targetdir()
		if is_plat("windows") then
			for i, v in ipairs(linkdirs) do
				os.cp(path.join(v, "cudadevrt.lib"), bin_dir)
			end
		end
		-- TODO: linux
	end
end)
target_end()