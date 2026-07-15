target("lc-spirv-llvm")
set_basename("luisa-spirv-llvm")
_config_project({
    project_kind = "static"
})
add_deps("lc-vstl", "lc-runtime", "spirv-tools")
add_files('*.cpp')

on_load(function(target)
    -- LLVM build root from lc_llvm_path config (set in scripts/options.lua)
    local lc_llvm_path = get_config("lc_llvm_path")
    -- Try build/debug or build/release first, fall back to lc_llvm_path directly
    local llvm_build_dir
    local debug_dir = path.join(lc_llvm_path, "build", "debug")
    local release_dir = path.join(lc_llvm_path, "build", "release")
    if is_mode("debug") and os.exists(debug_dir) then
        llvm_build_dir = debug_dir
    else
        llvm_build_dir = release_dir
    end

    target:add("includedirs", path.join(llvm_build_dir, "include"), {
        public = true
    })
    -- Also add LLVM source include dir (for in-tree builds where headers are split)
    local llvm_src_include = path.join(lc_llvm_path, "llvm", "include")
    if os.exists(llvm_src_include) then
        target:add("includedirs", llvm_src_include, {
            public = true
        })
    end
    local clang_src_include = path.join(lc_llvm_path, "clang", "include")
    if os.exists(clang_src_include) then
        target:add("includedirs", clang_src_include, {
            public = true
        })
    end
    target:add("includedirs", path.join(os.scriptdir(), ".."), {
        public = true
    })
    target:add("includedirs", path.join(os.scriptdir(), "..", "..", "..", "..", "include"), {
        public = true
    })
    target:add("linkdirs", path.join(llvm_build_dir, "lib"))

    local libs = {}
    for _, filepath in ipairs(os.files(path.join(llvm_build_dir, "lib/LLVM*.lib"))) do
        local basename = path.basename(filepath)
        if basename ~= "LLVM-C" then
            table.insert(libs, basename)
        end
    end
    target:add("links", libs)

    if is_plat("windows") then
        target:add("syslinks", "Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", "Ws2_32", "ntdll")
    elseif is_plat("linux") then
        target:add("syslinks", "uuid")
    elseif is_plat("macosx") then
        target:add("frameworks", "CoreFoundation")
    end
end)

target_end()
