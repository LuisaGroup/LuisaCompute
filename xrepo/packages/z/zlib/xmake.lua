package("zlib")
    set_homepage("http://zlib.net/")
    set_description("zlib 1.2.13 is a general purpose data compression library.")
    set_license("ZLIB License")
    
    add_versions("1.2.8-skr", "e3ed8a269df53e9a8530057f4949f48885f746ae0fe9d72e5b4c616833494dc2")

    --[[
    if (is_plat("windows")) then 
        add_links("zlibstatic")
    end
    if (is_plat("macosx")) then 
        add_links("z")
    end
    ]]--

    on_install(function (package)
        os.mkdir(package:installdir())
        os.cp(path.join(os.scriptdir(), "port", "zlib"), ".")
        os.cp(path.join(os.scriptdir(), "port", "xmake.lua"), "xmake.lua")

        if (is_plat("windows")) then 
            os.trycp("zlib/lib/windows/x64/*", package:installdir("lib"))
        end
        if (is_plat("macosx")) then 
            os.trycp("zlib/lib/macos/x86_64/*", package:installdir("lib"))
        end

        local configs = {}
        import("package.tools.xmake").install(package, configs)
    end)