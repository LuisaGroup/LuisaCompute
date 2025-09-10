if not is_mode("debug") then
    target("lc-clangcxx")
    _config_project({
        project_kind = "shared"
    })
    set_pcxxheader("src/lc_clangcxx_pch.h")
    add_headerfiles("../../include/luisa/clangcxx/**.h")
    add_files("src/**.cpp")
    on_load(function(target, opt)
        local libs = {}
        local llvm_path = get_config("llvm_path")
        if (not llvm_path) or (llvm_path == "") then
            llvm_path = path.join(os.scriptdir(), "llvm")
        end
        local p = path.join(llvm_path, "lib/*.lib")
        target:add("linkdirs", path.join(llvm_path, "lib"))
        target:add("includedirs", path.join(llvm_path, "include"))
        for __, filepath in ipairs(os.files(p)) do
            local basename = path.basename(filepath)
            table.insert(libs, basename)
        end
        target:add("links", libs)
        target:add("defines", "LC_CLANGCXX_EXPORT_DLL")
        target:add("deps", "lc-core", "lc-runtime", "lc-vstl")
        if is_plat("windows") then
            target:add("syslinks", "Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", 'Ws2_32', 'ntdll', {
                public = true
            })
        elseif is_plat("linux") then
            target:add("syslinks", "uuid")
        elseif is_plat("macosx") then
            target:add("frameworks", "CoreFoundation")
        end
        if is_mode("release") then
            target:add("defines", "LC_CLANGCXX_ENABLE_COMMENT=0")
        else
            target:add("defines", "LC_CLANGCXX_ENABLE_COMMENT=1")
        end

    end)
    target_end()
end
