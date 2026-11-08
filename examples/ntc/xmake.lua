-- LuisaCompute NTC (RTX Neural Texture Compression) xmake configuration

target("luisa-ntc")
    add_deps("lc-backends-dummy", {inherit = false, links = false})
    _config_project({
        project_kind = "static"
    })
    add_files("src/core/*.cpp", "src/infer/*.cpp", "src/bcn/*.cpp", "src/train/*.cpp")
    add_includedirs("include", "include/libntc", "$(projectdir)/examples", "$(projectdir)/src/ext/yyjson/src", "$(projectdir)/src/ext/stb", {public = true})
    add_deps("lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
    if has_config("lc_yyjson_use_xrepo") then
        add_packages("yyjson", {public = true})
    else
        add_deps("lc-yyjson")
    end
target_end()

local function ntc_binary(name, source, is_gui)
    target(name)
        add_deps("lc-backends-dummy", {inherit = false, links = false})
        _config_project({
            project_kind = "binary"
        })
        add_files(source)
        add_deps("luisa-ntc", "lc-runtime", "lc-dsl", "lc-vstl", "stb-image")
        add_includedirs("include", "include/libntc", "$(projectdir)/examples", "$(projectdir)/src/ext/yyjson/src", "$(projectdir)/src/ext/stb")
        if is_gui and has_config("lc_enable_gui") then
            add_deps("lc-gui")
            add_defines("LUISA_ENABLE_GUI")
        end
    target_end()
end

ntc_binary("luisa_ntc_cli", "tools/luisa_ntc_cli.cpp", false)
ntc_binary("luisa_ntc_bench", "tools/luisa_ntc_bench.cpp", false)
ntc_binary("luisa_ntc_test", "tools/luisa_ntc_test.cpp", false)
ntc_binary("luisa_ntc_viewer", "tools/luisa_ntc_viewer.cpp", true)
