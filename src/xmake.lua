table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("ext/volk", "ext/stb")
-- ext
lc_eastl_enable_custom_malloc = has_config("lc_enable_custom_malloc")
lc_eastl_enable_mimalloc = has_config("lc_enable_mimalloc")
includes("ext/EASTL")
if not has_config("lc_spdlog_use_xrepo") then
    includes("ext/spdlog")
end
if not has_config("lc_reproc_use_xrepo") then
    includes("ext/reproc")
end
if not has_config("lc_lmdb_use_xrepo") then
    includes("ext/liblmdb")
end
lc_eastl_enable_mimalloc = nil
lc_eastl_enable_custom_malloc = nil
-- yyjson
if not has_config("lc_yyjson_use_xrepo") then
    target("lc-yyjson")
    _config_project({
        project_kind = "static"
    })
    on_load(function(target)
        local src_path = path.join(os.scriptdir(), "ext/yyjson/src")
        target:add("files", path.join(src_path, "yyjson.c"))
        target:add("includedirs", src_path, {
            public = true
        })
        target:add("cxflags", "/utf-8", {
            tools = "cl"
        })
    end)
    target_end()
end
table.remove(_config_rules, rename_rule_idx)
includes("core", "vstl", "runtime")
includes("rust", "ir")
if has_config("lc_enable_xir") then
    includes("xir")
end
if has_config("lc_enable_osl") then
    includes("osl")
end
if has_config("lc_enable_dsl") then
    includes("dsl")
    includes("coro")
end
if has_config("lc_enable_gui") then
    includes("gui")
end
if has_config("_lc_enable_py") then
    includes("py")
end
includes("backends")
if has_config("lc_enable_tests") then
    includes("tests")
end
if has_config("lc_enable_clangcxx") then
    includes("clangcxx")
end

-- includes("tensor")
