target("lc-clangcxx")
    set_runtimes("MD")
    _config_project({
        project_kind = "shared",
        batch_size = 4
    })
    add_defines("LC_CLANGCXX_EXPORT_DLL")
    add_deps("lc-core", "lc-runtime")
    if is_plat("windows") then
        add_links("Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", {public = true})
    elseif is_plat("linux") then
        add_syslinks("uuid")
    elseif is_plat("macosx") then
        add_frameworks("CoreFoundation")
    end
    -- set_pcxxheader("pch.h")
    add_headerfiles("../../include/luisa/clangcxx/**.h")
    add_files("**.cpp")
    add_linkdirs("llvm/lib")
    add_includedirs("llvm/include")
    on_load(function (target, opt)
        local libs = {}
        local p = "$(scriptdir)/llvm/lib/*.lib"
        for __, filepath in ipairs(os.files(p)) do
            local basename = path.basename(filepath)
            table.insert(libs, basename)
        end
        target:add("links", libs)
    end)

