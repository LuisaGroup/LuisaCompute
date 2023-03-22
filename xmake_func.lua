-- Global config
rule("basic_settings")
on_config(function(target)
	if (not is_mode("debug")) then
		local _, cc = target:tool("cxx")
		local _, ld = target:tool("ld")
		if cc == "gcc" or cc == "gxx" then
			target:add("cxflags", "-flto")
			-- elseif (cc == "clang" or cc == "clangxx") then
			-- 	target:add("cxflags", "-flto=thin")
		end
		local function _add_link(...)
			target:add("ldflags", ...)
			target:add("shflags", ...)
		end
		if ld == "link" then
			_add_link("/INCREMENTAL:NO", "/LTCG")
			-- elseif (ld == "clang" or ld == "clangxx") then
			-- 	_add_link("-flto=thin")
		elseif ld == "gcc" or ld == "gxx" then
			_add_link("-flto")
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
		target:add("cxflags", "/GL", "/Zc:preprocessor", {
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
_config_rules = {"basic_settings"}
_disable_unity_build = not get_config("enable_unity_build")
_configs = {
	use_simd = UseSIMD
	-- enable_mimalloc=true
	-- enable_eastl_rtti=true
}
