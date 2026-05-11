target("lc-spirv")
set_basename("luisa-spirv")
_config_project({
    project_kind = "static",
    batch_size = 2
})
add_deps("lc-vstl", 'lc-runtime', 'lc-glslang')
on_load(function(target)
    if not target:is_plat("windows") then
        target:add("cxflags", "-fms-extensions", {
            tools = {"clang"},
            public = true
        })
    end
end)
add_files("*.cpp")
lc_set_pcxxheader("lc_spirv_pch.h")
add_headerfiles("*.h")
target_end()
