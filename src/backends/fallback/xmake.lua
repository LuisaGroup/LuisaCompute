local fallback_xmake_dir = os.scriptdir()

-- Resolve the Embree library and binary directories from an installation root.
-- A standard Embree install exposes lib/ (import libs + shared libs on Linux)
-- and bin/ (DLLs on Windows). The Embree distribution bundled inside
-- src/ext/HIPRT/contrib/embree uses a platform-specific layout instead:
-- include/ (headers), win/ (Windows .lib + .dll), linux/ (Linux .so).
local function embree_resolve_dirs(embree_root)
    local function pick(subdir, plat_subdir)
        local dir = path.join(embree_root, subdir)
        if not os.isdir(dir) then
            dir = path.join(embree_root, plat_subdir)
        end
        return dir
    end
    local lib_dir, bin_dir
    if is_plat("windows") then
        lib_dir = pick("lib", "win")
        bin_dir = pick("bin", "win")
    elseif is_plat("linux") then
        lib_dir = pick("lib", "linux")
        bin_dir = pick("bin", "linux")
    else
        lib_dir = path.join(embree_root, "lib")
        bin_dir = path.join(embree_root, "bin")
    end
    return lib_dir, bin_dir
end

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
    local embree_lib_dir = embree_resolve_dirs(lc_embree_path)
    target:add("linkdirs", path.join(lc_llvm_path, "lib"), embree_lib_dir)
    target:add("includedirs", path.join(lc_llvm_path, "include"), path.join(lc_embree_path, "include"))
    target:add("links", "embree4", "tbb12")
    -- Detect the Embree major version from rtcore_config.h so the
    -- LUISA_COMPUTE_FALLBACK_EMBREE_VERSION define matches the linked library.
    local embree_version = 4
    local embree_include = path.join(lc_embree_path, "include")
    for _, rel in ipairs({"embree4/rtcore_config.h", "embree3/rtcore_config.h"}) do
        local config_file = path.join(embree_include, rel)
        if os.isfile(config_file) then
            local content = io.readfile(config_file)
            local major = content:match("RTC_VERSION_MAJOR%s+(%d+)")
            if major then
                embree_version = tonumber(major)
                break
            end
        end
    end
    target:add("defines", "LUISA_COMPUTE_FALLBACK_EMBREE_VERSION=" .. embree_version)
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
    local _, embree_bin_dir = embree_resolve_dirs(lc_embree_path)
    local dst_path = target:targetdir()
    for __, filepath in ipairs(os.files(path.join(embree_bin_dir, "*.dll"))) do
        copy(filepath, path.join(dst_path, path.filename(filepath)))
    end
    for __, filepath in ipairs(os.files(path.join(lc_llvm_path, "bin/*.dll"))) do
        copy(filepath, path.join(dst_path, path.filename(filepath)))
    end
    ::END::
end)
add_files("*.cpp")
add_files("../common/default_binary_io.cpp")
add_files(
    "../common/llvm_native_math.cpp",
    "../common/llvm_native_math_atan2.cpp",
    "../common/llvm_native_math_fast_exp_log.cpp",
    "../common/llvm_native_math_fast_inverse_trig.cpp",
    "../common/llvm_native_math_fast_trig.cpp",
    "../common/llvm_native_math_hyperbolic.cpp",
    "../common/llvm_native_math_pow.cpp",
    "../common/llvm_native_math_precise_exp_log.cpp",
    "../common/llvm_native_math_precise_inverse_trig.cpp",
    "../common/llvm_native_math_precise_trig.cpp",
    "../common/llvm_native_math_range_reduction.cpp")
add_deps("lc-runtime")
target_end()
