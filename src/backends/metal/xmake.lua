target("lc-backend-metal-builtin")
set_kind('static')
add_rules("utils.bin2obj", {
    extensions = '.metal'
})
add_files('metal_builtin/*.metal')
target_end()

target("lc-backend-metal")
set_basename("luisa-backend-metal")
_config_project({
    project_kind = "shared",
    batch_size = 0
})
add_deps("lc-runtime")
add_headerfiles("*.h")
add_files("*.mm")
add_files("metal_builtin/*.metal", "metal-tex-compress/*.patched.metal", {rules = "utils.bin2obj"})

on_load(function(target)
    local src_path = os.scriptdir()
    for _, filepath in ipairs(os.files(path.join(src_path, "*.cpp"))) do
        local file_name = path.filename(filepath)
        if file_name ~= "metal_builtin_embedded.cpp" then
            target:add("files", filepath)
        end
    end
    target:add("files", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.cpp")))
end)

add_frameworks("Foundation", "Metal", "QuartzCore", "AppKit")
add_syslinks("compression")
add_defines('LUISA_BIN_2_OBJ')
target_end()
