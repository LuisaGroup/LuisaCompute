-- export config.h and xmake_config.lua
if ExportConfig then
	function lc_add_defines(...)
		if ExportConfig then
			local args = {...};
			local lastArg = args[table.getn(args)]
			if type(lastArg) == "table" and lastArg.public then
				for i, v in ipairs(args) do
					if type(v) == "string" then
						add_values("lc_defines", v)
					end
				end
			end
		end
		add_defines(...)
	end
	function lc_add_includedirs(...)
		if ExportConfig then
			local args = {...};
			local lastArg = args[table.getn(args)]
			if type(lastArg) == "table" and lastArg.public then
				for i, v in ipairs(args) do
					if type(v) == "string" then
						local absPath = path.relative(v, os.projectdir()):gsub('\\', '/')
						add_values("lc_includedir", absPath)
					end
				end
			end
		end
		add_includedirs(...)
	end
	rule("export_define")
	after_build(function(target)
		local defines = target:values("lc_defines")
		local incDirs = target:values("lc_includedir")
		if defines == nil and incDirs == nil then
			return
		end

		local file = io.open("config/xmake_config.lua", "a")
		function print_comment()
			file:write("-- " .. target:name() .. '\n')
		end
		if not file then
			return
		end
		local commentWrited = false
		local defineStr = "add_defines("
		if defines ~= nil then
			for i, v in ipairs(defines) do
				defineStr = defineStr .. '"' .. v .. '",'
			end
			defineStr = defineStr .. "{public=isPublic})\n"
			commentWrited = true
			print_comment()
			file:write(defineStr)
		end
		if incDirs ~= nil then
			local incStr = "add_includedirs("
			for i, v in ipairs(incDirs) do
				incStr = incStr .. 'rootDir.."/' .. v .. '",'
			end
			if not commentWrited then
				print_comment()
			end
			incStr = incStr .. "{public=isPublic})\n"
			file:write(incStr)
		end
		file:close()
	end)
	rule_end()
	rule("export_define_project")
	set_kind("project")
	before_build(function()
		io.writefile("config/xmake_config.lua", "function add_lc_includes(rootDir, isPublic)\n")
	end)
	after_build(function()
		local file = io.open("config/xmake_config.lua", "a")
		if file then
			file:write("end")
			file:close()
		end
	end)
	rule_end()
end
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
if ExportConfig then
	table.insert(_configs.events, function(config)
		add_rules("export_define")
	end);
	_configs.override_add_defines = lc_add_defines
	_configs.override_add_includedirs = lc_add_includedirs
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
			_add_link_flags("/INCREMENTAL:NO", "/LTCG", "/OPT:REF", "/OPT:ICF")
		else
			_add_link_flags("/LTCG:incremental")
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
