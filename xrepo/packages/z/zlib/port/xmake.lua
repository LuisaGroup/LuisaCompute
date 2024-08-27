set_languages("c11", "cxx17")

target("zlib")
    set_kind("headeronly")
    add_includedirs("zlib/include", {public=true})
    add_headerfiles("zlib/include/(**.h)")
    if (is_os("windows")) then 
        add_linkdirs("zlib/lib/windows/x64", {public=true})
        add_links("zlibstatic", {public=true})
    end
    if (is_os("macosx")) then 
        add_linkdirs("zlib/lib/macos/x86_64", {public=true})
        add_links("z", {public=true})
    end