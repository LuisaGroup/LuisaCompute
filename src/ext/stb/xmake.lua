target("stb-image")
_config_project({
    project_kind = "static"
})
add_headerfiles("stb/**.h")
add_files("stb.c")
add_includedirs(".", {
    public = true
})
-- Fallback for <stb/stb_image_resize2.h>: upstream stb moved the header out of
-- the repo (see compat/stb/stb_image_resize2.h). Searched after the submodule.
add_includedirs("compat", {
    public = true
})
target_end()