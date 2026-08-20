target("example_gdeflate")
add_deps("lc-backends-dummy", {inherit = false, links = false})
_config_project({
    project_kind = "binary"
})
add_files("*.cpp")
add_includedirs("$(projectdir)/src/tests/", "$(projectdir)/src/tests/common/", "$(projectdir)/examples/")
add_deps("lc-runtime", "lc-dsl", "lc-vstl")
target_end()
