target("lc-coro")
set_basename("luisa-coro")
_config_project({
    project_kind = "shared",
    batch_size = 0
})
add_deps("lc-dsl")
add_headerfiles("../../include/luisa/coro/**.h")
add_files("**.cpp", "schedulers/**.cpp")
add_defines("LUISA_CORO_EXPORT_DLL", {
    private = true
})
target_end()
