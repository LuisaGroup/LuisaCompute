enable_mimalloc = get_config("enable_mimalloc")
table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("ext/EASTL")
includes("ext/spdlog")
includes("ext/reproc")
table.remove(_config_rules, rename_rule_idx)
includes("core")
includes("vstl")
includes("ast")
includes("runtime")
includes("osl")
if get_config("enable_dsl") then
	includes("dsl")
end
if get_config("enable_gui") then
	includes("gui")
end
if get_config("_lc_enable_py") then
	includes("py")
end
includes("backends")
if get_config("enable_tests") then
	includes("tests")
end
if get_config("_lc_enable_rust") then
	includes("rust")
end
if get_config("enable_ir") then
	includes("ir")
end
if get_config("enable_api") then
	includes("api")
end