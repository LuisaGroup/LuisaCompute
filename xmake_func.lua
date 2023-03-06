-- Global config
_configs = {
	languages = {"clatest", "cxx20"},
	enable_unitybuild = UseUnityBuild,
	events = {}
}

if is_mode("debug") then
	_configs.win_runtime = "MDd"
	table.insert(_configs.events, function(config)
		set_optimize("none")
		set_warnings("none")
		add_cxflags("/GS", "/Gd", {
			tools = {"clang_cl", "cl"}
		})
		add_cxflags("/Zc:preprocessor", {
			tools = "cl"
		});
	end)
else
	_configs.win_runtime = "MD"
	table.insert(_configs.events, function(config)
		set_optimize("fastest")
		set_warnings("none")
		add_cxflags("/Oy", "/GS-", "/Gd", "/Oi", "/Ot", {
			tools = {"clang_cl", "cl"}
		})
		add_cxflags("/GL", "/Zc:preprocessor", {
			tools = "cl"
		})
	end)
end

if UseSIMD then
	table.insert(_configs.events, function(config)
		add_vectorexts("sse", "sse2")
	end);
end
table.insert(_configs.events, function(config)
	function _add_link_flags(...)
		if config.project_kind == "shared" then
			add_shflags(...)
		elseif config.project_kind == "binary" then
			add_ldflags(...)
		else
			add_arflags(...)
		end
	end
	if is_plat("windows") then
		if is_mode("release") then
			_add_link_flags("/INCREMENTAL:NO", "/LTCG:INCREMENTAL", "/OPT:REF", "/OPT:ICF")
		else
			_add_link_flags("/LTCG:OFF")
		end
	end

	if config.no_rtti then
		add_cxflags("/GR-", {
			tools = {"clang_cl", "cl"}
		})
		add_cxflags("-fno-rtti", "-fno-rtti-data", {
			tools = {"clang", "gcc"}
		})
	end
end)
