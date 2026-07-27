target("lc-backend-fallback")
set_basename("luisa-backend-fallback")
_config_project({
    project_kind = "shared",
    batch_size = 8
})
on_load(function(target, opt)
    local libs = {}
    local lc_llvm_path = get_config("lc_llvm_path")
    local lc_embree_path = get_config("lc_embree_path")
    target:add("linkdirs", path.join(lc_llvm_path, "lib"), path.join(lc_embree_path, "lib"))
    target:add("includedirs", path.join(lc_llvm_path, "include"), path.join(lc_embree_path, "include"))
    target:add("links", "embree4", "tbb12")
    for __, filepath in ipairs(os.files(path.join(lc_llvm_path, "lib/*.lib"))) do
        local basename = path.basename(filepath)
        if basename:match("LLVM") ~= nil and basename ~= "LLVM-C" then
            table.insert(libs, basename)
        end
    end
    target:add("links", libs)
    if is_plat("windows") then
        target:add("syslinks", "Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", 'Ws2_32', 'ntdll', {
            public = true
        })
    elseif is_plat("linux") then
        target:add("syslinks", "uuid")
    elseif is_plat("macosx") then
        target:add("frameworks", "CoreFoundation")
    end
    target:add("defines", "LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN")
    target:add("deps", "lc-vulkan-swapchain", "lc-volk")
end)
after_build(function(target)
    if not is_plat("windows") then
        goto END
    end
    local function copy(src_path, dst_path)
        os.cp(src_path, dst_path, {
            copy_if_different = true,
            async = true,
            detach = true
        })
    end
    local lc_llvm_path = get_config("lc_llvm_path")
    local lc_embree_path = get_config("lc_embree_path")
    local dst_path = target:targetdir()
    for __, filepath in ipairs(os.files(path.join(lc_embree_path, "bin/*.dll"))) do
        copy(filepath, path.join(dst_path, path.filename(filepath)))
    end
    for __, filepath in ipairs(os.files(path.join(lc_llvm_path, "bin/*.dll"))) do
        copy(filepath, path.join(dst_path, path.filename(filepath)))
    end
    ::END::
end)
add_files("*.cpp")
add_files("../common/default_binary_io.cpp")
add_deps("lc-runtime")
target_end()
