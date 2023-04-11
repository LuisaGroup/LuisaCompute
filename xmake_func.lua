-- Global config
rule("lc_vulkan")
on_load(function(target)
	local vk_path = target:values("vk_path")
	local macro_value = target:values("vulkan_macro")
	local enable_swapchain = target:values("enable_swapchain")
	if macro_value and type(macro_value) == "string" then
		target:add("defines", macro_value)
	end
	target:add("linkdirs", path.join(vk_path, "Lib"))
	target:add("links", "vulkan-1")
	target:add("includedirs", path.join(vk_path, "Include"))
	if enable_swapchain then
		target:add("files", path.join(os.scriptdir(), "src/backends/common/vulkan_swapchain.cpp"))
	end
end)
rule_end()
rule("lc_basic_settings")
on_config(function(target)
	local _, cc = target:tool("cxx")
	if is_plat("linux") then
		-- Linux should use -stdlib=libc++
		-- https://github.com/LuisaGroup/LuisaCompute/issues/58
		if (cc == "clang" or cc == "clangxx") then
			target:add("cxflags", "-stdlib=libc++", {
				force = true
			})
		end
	end
end)
on_load(function(target)
	local _get_or = function(name, default_value)
		local v = target:values(name)
		if v == nil then
			return default_value
		end
		return v
	end
	local project_kind = _get_or("project_kind", "phony")
	target:set("kind", project_kind)
	local c_standard = target:values("c_standard")
	local cxx_standard = target:values("cxx_standard")
	if type(c_standard) == "string" and type(cxx_standard) == "string" then
		target:set("languages", c_standard, cxx_standard)
	else
		target:set("languages", "clatest", "cxx20")
	end

	local enable_exception = _get_or("enable_exception", nil)
	if enable_exception then
		target:set("exceptions", "cxx")
	else
		target:set("exceptions", "no-cxx")
	end
	if is_mode("debug") then
		target:set("runtimes", "MDd")
		target:set("optimize", "none")
		target:set("warnings", "none")
		target:add("cxflags", "/GS", "/Gd", {
			tools = {"clang_cl", "cl"}
		})
		target:add("cxflags", "/Zc:preprocessor", {
			tools = "cl"
		});
	else
		target:set("runtimes", "MD")
		target:set("optimize", "aggressive")
		target:set("warnings", "none")
		target:add("cxflags", "/GS-", "/Gd", {
			tools = {"clang_cl", "cl"}
		})
		target:add("cxflags", "/Zc:preprocessor", {
			tools = "cl"
		})
	end
	if _get_or("use_simd", false) then
		target:add("vectorexts", "sse", "sse2")
	end
	if _get_or("no_rtti", false) then
		target:add("cxflags", "/GR-", {
			tools = {"clang_cl", "cl"}
		})
		target:add("cxflags", "-fno-rtti", "-fno-rtti-data", {
			tools = {"clang", "gcc"}
		})
	end
end)
rule_end()
-- In-case of submod, when there is override rules, do not overload
if _config_rules == nil then
	_config_rules = {"lc_basic_settings"}
end
if _disable_unity_build == nil then
	_disable_unity_build = not get_config("enable_unity_build")
end

if _configs == nil then
	_configs = {}
end
_configs["use_simd"] = LCUseSIMD
