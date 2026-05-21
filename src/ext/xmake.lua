table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("volk", "stb")
-- ext
lc_eastl_enable_custom_malloc = has_config("lc_enable_custom_malloc")
lc_eastl_enable_mimalloc = has_config("lc_enable_mimalloc")
includes("EASTL")
if has_config('lc_vk_backend_use_xir_spirv') then
    includes("glslang")
end

if not has_config("lc_spdlog_use_xrepo") then
    includes("spdlog")
end
if not has_config("lc_reproc_use_xrepo") then
    includes("reproc")
end
if not has_config("lc_lmdb_use_xrepo") then
    includes("liblmdb")
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
        local src_path = path.join(os.scriptdir(), "yyjson/src")
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
