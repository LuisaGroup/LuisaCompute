if LCUseMimalloc then
	_configs.enable_mimalloc = true
end
table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("ext/EASTL")
_configs.enable_mimalloc = nil
includes("ext/spdlog")
table.remove(_config_rules, rename_rule_idx)
includes("core")
includes("vstl")
includes("ast")
includes("runtime")
if LCEnableDSL then
	includes("dsl")
end
if LCEnableGUI then
	includes("gui")
end
if LCEnablePython then
	includes("py")
end
includes("backends")
if LCEnableTest then
	includes("tests")
end
if LCEnableRust then
	includes("rust")
end
if LCEnableIR then
	includes("ir")
end
if LCEnableAPI then
	includes("api")
end
