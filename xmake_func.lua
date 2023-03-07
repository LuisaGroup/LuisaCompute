-- Global config
rule("basic_settings")
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
		target:set("exceptions", "cxx", "objc")
	else
		target:set("exceptions", "no-cxx", "no-objc")
	end
	if is_plat("windows") then
		local win_runtime = _get_or("win_runtime", nil)
		if win_runtime ~= nil then
			target:set("runtimes", win_runtime)
		else
			if is_mode("debug") then
				target:set("runtimes", "MDd")
			else
				target:set("runtimes", "MD")
			end
		end
	end
	if is_mode("debug") then
		target:set("optimize", "none")
		target:set("warnings", "none")
		target:add("cxflags", "/GS", "/Gd", {
			tools = {"clang_cl", "cl"}
		})
		target:add("cxflags", "/Zc:preprocessor", {
			tools = "cl"
		});
	else
		target:set("optimize", "fastest")
		target:set("warnings", "none")
		target:add("cxflags", "/Oy", "/GS-", "/Gd", "/Oi", "/Ot", {
			tools = {"clang_cl", "cl"}
		})
		target:add("cxflags", "/GL", "/Zc:preprocessor", {
			tools = "cl"
		})
	end
	if _get_or("use_simd", false) then
		target:add("vectorexts", "sse", "sse2")
	end
	local function _add_link_flags(...)
		if project_kind == "shared" then
			target:add("shflags", ...)
		elseif project_kind == "binary" then
			target:add("ldflags", ...)
		else
			target:add("arflags", ...)
		end
	end
	if is_plat("windows") then
		if is_mode("release") then
			_add_link_flags("/INCREMENTAL:NO", "/LTCG:INCREMENTAL", "/OPT:REF", "/OPT:ICF")
		else
			_add_link_flags("/LTCG:OFF")
		end
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
_configs = {
	use_simd = UseSIMD
	-- enable_mimalloc=true
	-- enable_eastl_rtti=true
}
