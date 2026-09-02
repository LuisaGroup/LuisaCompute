target("lc-tile")
set_basename("luisa-tile")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
add_deps("lc-core")
add_defines("LUISA_TILE_EXPORT_DLL")
add_headerfiles("../../include/luisa/tile/**.h", "../../include/luisa/tile.h")
add_files("*.cpp")
target_end()
