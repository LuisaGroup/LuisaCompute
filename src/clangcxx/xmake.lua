if not is_mode("debug") then
    target("lc-clangcxx")
    set_basename("luisa-clangcxx")
    _config_project({
        project_kind = "shared"
    })
    lc_set_pcxxheader("src/lc_clangcxx_pch.h")
    add_files("src/**.cpp")
    on_load(function(target, opt)
        target:add("headerfiles", path.normalize(path.join(os.scriptdir(), "../common/default_binary_io.h")))
        local libs = {}
        local lc_llvm_path = get_config("lc_llvm_path")
        if (not lc_llvm_path) or (lc_llvm_path == "") then
            lc_llvm_path = path.join(os.scriptdir(), "llvm")
        end
        -- Try build/debug or build/release first, fall back to lc_llvm_path directly
        local llvm_build_dir
        local debug_dir = path.join(lc_llvm_path, "build", "debug")
        local release_dir = path.join(lc_llvm_path, "build", "release")
        if is_mode("debug") and os.exists(debug_dir) then
            llvm_build_dir = debug_dir
        elseif is_mode("release") and os.exists(release_dir) then
            llvm_build_dir = release_dir
        else
            llvm_build_dir = lc_llvm_path
        end
        local p = path.join(llvm_build_dir, "lib/*.lib")
        target:add("linkdirs", path.join(llvm_build_dir, "lib"))
        target:add("includedirs", path.join(llvm_build_dir, "include"))
        -- Also add LLVM source include dir (for in-tree builds where headers are split)
        local llvm_src_include = path.join(lc_llvm_path, "llvm", "include")
        if os.exists(llvm_src_include) then
            target:add("includedirs", llvm_src_include)
        end
        local clang_src_include = path.join(lc_llvm_path, "clang", "include")
        if os.exists(clang_src_include) then
            target:add("includedirs", clang_src_include)
        end
        local black_list = {
            "lld",
        }
        for __, filepath in ipairs(os.files(p)) do
            local basename = path.basename(filepath)
            for _, v in ipairs(black_list) do
                if basename:match(v) ~= nil then
                    goto END_LOOP
                end
            end
            table.insert(libs, basename)
            ::END_LOOP::
        end
        target:add("links", libs)
        target:add("defines", "LUISA_CLANGCXX_EXPORT_DLL", 'CLANG_BUILD_STATIC')
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
