local function test_proj(name, callable)
    target("amd_" .. name)
    _config_project({
        project_kind = "binary"
    })
    add_files(name .. ".cpp")
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "lc-backends-dummy", "stb-image")
    add_deps("lc-gui")
    if callable then
        callable()
    end
    target_end()
end

test_proj("test_copy")
test_proj("test_compress_dstorage")
