_get_or = nil
if _configs ~= nil then
	_get_or = function(name, default_value)
		local v = _configs[name]
		if v ~= nil then
			return v
		end
		return default_value
	end
else
	_get_or = function(name, default_value)
		return default_value
	end
end
function _config_project(config)
	local proj_name = _get_or("project_name", config.project_name);
	target(proj_name)
	set_kind(_get_or("project_kind", config.project_kind))
	local langs = _get_or("languages", nil)
	if langs ~= nil then
		set_languages(langs)
	else
		set_languages("clagest", "cxx20")
	end
	local enable_unitybuild = _get_or("enable_unitybuild", true)
	if enable_unitybuild then
		local unityBuildBatch = (config.batch_size)
		if (unityBuildBatch ~= nil) and (unityBuildBatch > 1) then
			add_rules("c.unity_build", {
				batchsize = unityBuildBatch
			})
			add_rules("c++.unity_build", {
				batchsize = unityBuildBatch
			})
		end
	end
	local enable_exception = _get_or("enable_exception", true) and (config.enable_exception)
	if enable_exception then
		set_exceptions("cxx", "objc")
	else
		set_exceptions("no-cxx", "no-objc")
	end
	if is_plat("windows") then
		local win_runtime = _get_or("win_runtime", nil)
		if win_runtime ~= nil then
			set_runtimes(win_runtime)
		else
			if is_mode("debug") then
				set_runtimes("MDd")
			else
				set_runtimes("MD")
			end
		end
	end
	local events = _get_or("events", nil)
	if events ~= nil then
		for i, k in ipairs(events) do
			k(config)
		end
	end
end

--[[
_configs = {
    languages = {},
    project_name = "",
    project_kind = "",
    enable_unitybuild = true,
	enable_exception = false,
	enable_rtti = false,
    override_add_defines = add_defines,
    override_add_includedirs = add_includedirs,
    win_runtime = "MD",
    events = {function() end}
}
]]
