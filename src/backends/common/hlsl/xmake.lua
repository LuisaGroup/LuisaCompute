target("lc-hlsl-codegen")
set_basename("luisa-hlsl-codegen")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl")
on_load(function(target)
    if not target:is_plat("windows") then
        target:add("cxflags", "-fms-extensions", {
            tools = {"clang"},
            public = true
        })
    end
end)
add_files("*.cpp", 'codegen_utils/*.cpp')
lc_set_pcxxheader("lc_hlsl_pch.h")
add_headerfiles("*.h")
-- XMake embeds the canonical inputs directly into objects, so it never
-- consumes CMake's checked-in aggregate and tracks each input independently.
add_rules("utils.bin2obj", {extensions = {".bytes", ".dxil"}})
add_defines('LUISA_BIN_2_OBJ')
add_files('builtin/*.bytes', 'builtin/*.dxil')
target_end()
