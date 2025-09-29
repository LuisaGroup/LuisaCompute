target("lc-backend-metal")
_config_project({
    project_kind = "shared",
    batch_size = 0
})
add_deps("lc-runtime")
add_headerfiles("*.h")
add_files("*.cpp", "*.mm", "../common/default_binary_io.cpp")

add_frameworks("Foundation", "Metal", "QuartzCore", "AppKit")
add_syslinks("compression")

add_deps("lc_embed_codegen", {
    inherit = false,
    public = false
})
add_rules("lc_compile_codegen", {
    remove_ext = true,
    remove_slash_r = true,
    var_name_prefix = "luisa_compute_"
})
add_files("metal_builtin.lua")
target_end()
