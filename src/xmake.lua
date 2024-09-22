enable_mimalloc = get_config("enable_mimalloc")
enable_custom_malloc = get_config("enable_custom_malloc")
table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("ext/EASTL", "ext/spdlog", "ext/reproc", "ext/liblmdb", "ext/volk", "ext/stb")
table.remove(_config_rules, rename_rule_idx)
includes("core", "vstl", "ast", "runtime")
if get_config("enable_osl") then
    includes("osl")
end
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
if get_config("enable_clangcxx") then
    includes("clangcxx")
end

add_requires("zlib >=1.2.8-skr", {
    system = false
})