local fallback_xmake_dir = os.scriptdir()

target("lc-fallback-embed-device-lib")
set_basename("luisa-embed-device-lib")
set_kind("binary")
set_default(false)
set_policy("build.fence", true)
_config_project({
    project_kind = "binary"
})
add_files("../../../utils/embed_device_lib.cpp")
target_end()

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
    target:add("deps", "lc-vulkan-swapchain", "lc-volk",
               "lc-fallback-embed-device-lib")

    local generated_dir = path.join(
        target:autogendir(), "fallback_builtin")
    target:add("includedirs", generated_dir)
    target:add("files", path.join(
        generated_dir, "fallback_device_api_wrappers_embedded.cpp"), {
            always_added = true
        })
end)
before_build(function(target)
    import("core.project.depend")
    import("lib.detect.find_tool")

    local llvm_root = get_config("lc_llvm_path")
    if type(llvm_root) ~= "string" or llvm_root:len() == 0 then
        raise("lc-backend-fallback requires lc_llvm_path to name an LLVM installation.")
    end

    local executable_suffix = target:is_plat("windows") and ".exe" or ""
    local clang_tool_name = "clang++"
    local candidate_bin_dirs = {}
    if os.isfile(llvm_root) then
        table.insert(candidate_bin_dirs, path.directory(llvm_root))
    else
        for _, candidate in ipairs({
            path.join(llvm_root, "build", "release", "bin"),
            path.join(llvm_root, "build", "Release", "bin"),
            path.join(llvm_root, "build", "debug", "bin"),
            path.join(llvm_root, "build", "Debug", "bin"),
            path.join(llvm_root, "build", "bin"),
            path.join(llvm_root, "bin")
        }) do
            table.insert(candidate_bin_dirs, candidate)
        end
    end
    local function find_llvm_tool(name)
        for _, directory in ipairs(candidate_bin_dirs) do
            local candidate = path.join(
                directory, name .. executable_suffix)
            if os.isfile(candidate) then
                return candidate
            end
        end
        raise("Could not find " .. name .. " from the LLVM installation " ..
              "selected by lc_llvm_path=" .. llvm_root .. ". Refusing to " ..
              "use a tool from another LLVM package.")
    end

    local target_arch
    if target:is_arch("x86_64", "x64") then
        target_arch = "x86_64"
    elseif target:is_arch("arm64", "aarch64") then
        target_arch = "aarch64"
    else
        raise("Unsupported fallback target architecture: " .. target:arch())
    end
    local target_triple
    if target:is_plat("windows") then
        target_triple = target_arch .. "-pc-windows-msvc"
    elseif target:is_plat("macosx") then
        target_triple = target_arch .. "-apple-darwin"
    elseif target:is_plat("linux") then
        target_triple = target_arch .. "-unknown-linux-gnu"
    else
        raise("Unsupported fallback target platform: " .. target:plat())
    end

    local generated_dir = path.join(
        target:autogendir(), "fallback_builtin")
    local source = path.join(
        fallback_xmake_dir,
        "fallback_builtin/fallback_device_api_wrappers.cpp")
    local api_header = path.join(
        fallback_xmake_dir, "fallback_device_api.h")
    local generator = path.join(
        fallback_xmake_dir,
        "fallback_builtin/generate_and_embed_fallback_device_lib.py")
    local output_ll = path.join(
        generated_dir, "fallback_device_api_wrappers.ll")
    local output_bc = path.join(
        generated_dir, "fallback_device_api_wrappers.bc")
    local output_symbol_map = path.join(
        generated_dir, "fallback_device_api_map_symbols.generated.inl.h")
    local embedded_cpp = path.join(
        generated_dir, "fallback_device_api_wrappers_embedded.cpp")
    local embedded_h = path.join(
        generated_dir, "fallback_device_api_wrappers_embedded.h")
    local embedder = target:dep(
        "lc-fallback-embed-device-lib"):targetfile()
    local python = find_tool("python3") or find_tool("python")
    assert(python, "Python is required to generate fallback device bitcode.")

    depend.on_changed(function()
        os.mkdir(generated_dir)
        os.vrunv(python.program, {
            generator,
            "--clang", find_llvm_tool(clang_tool_name),
            "--llvm-as", find_llvm_tool("llvm-as"),
            "--source", source,
            "--target", target_triple,
            "--deployment-target",
            target:is_plat("macosx") and
                (get_config("target_minver") or "11.0") or "",
            "--output-ll", output_ll,
            "--output-bc", output_bc,
            "--output-symbol-map", output_symbol_map
        })
        os.vrunv(embedder, {
            output_bc,
            "--unsigned", "--prefix", "luisa_compute_",
            "-o", embedded_cpp, "-h", embedded_h
        })
    end, {
        dependfile = target:dependfile(embedded_cpp),
        files = {generator, source, api_header, embedder},
        changed = target:is_rebuilt() or
                  not os.isfile(output_ll) or
                  not os.isfile(output_bc) or
                  not os.isfile(output_symbol_map) or
                  not os.isfile(embedded_cpp) or
                  not os.isfile(embedded_h)
    })
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
