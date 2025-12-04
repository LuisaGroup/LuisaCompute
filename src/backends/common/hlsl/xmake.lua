target("lc-hlsl-codegen")
set_basename("luisa-hlsl-codegen")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl")
add_deps("lc_embed_codegen", {
    inherit = false
})
on_load(function(target)
    if not target:is_plat("windows") then
        target:add("cxflags", "-fms-extensions", {
            tools = {"clang"},
            public = true
        })
    end
end)
add_files("*.cpp")
set_pcxxheader("lc_hlsl_pch.h")
add_headerfiles("*.h")
add_rules("lc_compile_codegen")
add_files("builtin.lua")
target_end()
