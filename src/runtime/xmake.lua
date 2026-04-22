target("lc-runtime")
set_basename("luisa-runtime")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
add_deps("lc-core", "lc-vstl")
set_pcxxheader("lc_runtime_pch.h")
add_defines("LUISA_RUNTIME_EXPORT_DLL", "LUISA_AST_EXPORT_DLL")
add_headerfiles("../../include/luisa/runtime/**.h", "../../include/luisa/ast/**.h")
on_load(function(target)
    target:add("files", path.absolute("../ast/*.cpp", os.scriptdir()), path.join(os.scriptdir(), "**.cpp"))
end)
target_end()
