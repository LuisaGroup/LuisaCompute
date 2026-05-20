target("lc-spirv")
set_basename("luisa-spirv")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl", 'lc-runtime', 'lc-glslang', 'spirv-tools')
on_load(function(target)
    if not target:is_plat("windows") then
        target:add("cxflags", "-fms-extensions", {
            tools = {"clang"},
            public = true
        })
    end
end)
add_files("spirv_codegen/*.cpp")
lc_set_pcxxheader("spirv_codegen/lc_spirv_pch.h")
add_headerfiles("*.h")
add_includedirs('.', {
    public = true
})
target_end()
