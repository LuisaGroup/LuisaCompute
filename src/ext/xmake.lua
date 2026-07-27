table.insert(_config_rules, "lc-rename-ext")
local rename_rule_idx = table.getn(_config_rules)
includes("volk", "stb")
-- ext
lc_eastl_enable_custom_malloc = has_config("lc_enable_custom_malloc")
lc_eastl_enable_mimalloc = has_config("lc_enable_mimalloc")
includes("EASTL")
local need_spv_codegen = has_config("lc_vk_backend") and
                         (has_config("lc_vk_backend_use_xir_spirv") or
                          has_config("lc_vk_backend_use_ast_llvm_spirv"))
if need_spv_codegen then
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
-- The HLSL validation test compiles DXC output to SPIR-V and validates it with
-- SPIRV-Tools even when neither of the optional Vulkan SPIR-V code generators
-- is enabled. Keep that test dependency aligned with src/tests/xmake.lua.
local need_spv_tools = need_spv_codegen or
                       (has_config("lc_enable_tests") and
                        (has_config("lc_vk_backend") or
                         has_config("lc_dx_backend")))
if need_spv_tools then
    target('spirv-headers')
    set_kind('headeronly')
    add_includedirs("spirv-headers/include", "spirv-headers/include/spirv/unified1", {
        public = true
    })
    target_end()

    includes("SPIRV-Tools")
    -- SPIRV-Tools' broad source/*.cpp glob also picks up the optional
    -- mimalloc override, whose header is intentionally unavailable when the
    -- project allocator is disabled. CMake only compiles this source when
    -- SPIRV_TOOLS_USE_MIMALLOC is enabled; mirror that default here.
    target("spirv-tools")
    remove_files("SPIRV-Tools/source/mimalloc.cpp")
    target_end()
end

table.remove(_config_rules, rename_rule_idx)
