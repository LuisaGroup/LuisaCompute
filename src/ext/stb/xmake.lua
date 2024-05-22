target("stb-image")
_config_project({
    project_kind = "static"
})
add_headerfiles("stb/**.h")
add_files("stb.c")
add_includedirs(".", {
    public = true
})
target_end()