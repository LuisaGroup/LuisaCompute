option("_lc_enable_py", {
    default = false,
    showmenu = false
})

option("_lc_check_env")
set_showmenu(false)
set_default(false)
after_check(function(option)
    if not is_arch("x64", "x86_64", "arm64") then
        option:set_value(false)
        utils.error("Illegal environment. Please check your compiler, architecture or platform.")
        return
    end
    if not (is_mode("debug") or is_mode("release") or is_mode("releasedbg")) then
        option:set_value(false)
        utils.error("Illegal mode. set mode to 'release', 'debug' or 'releasedbg'.")
        return
    end
    option:set_value(true)
end)
option_end()

option("_lc_bin_dir")
set_default(false)
set_showmenu(false)
add_deps("lc_dx_backend", "lc_vk_backend", "lc_cuda_backend", "lc_metal_backend", "lc_enable_tests", "lc_py_include",
    "lc_cuda_ext_lcub", "lc_enable_dsl", "lc_enable_gui", "lc_bin_dir", "lc_dx_cuda_interop", "lc_vk_cuda_interop",
    "_lc_enable_py", "lc_enable_py", "lc_enable_xir", "lc_fallback_backend", "lc_llvm_path", "lc_embree_path")
before_check(function(option)
    if path.absolute(path.join(os.projectdir(), "scripts")) == path.absolute(os.scriptdir()) then
        local v = import("options", {
            try = true,
            anonymous = true
        })
        if v then
            local opt = v.get_options
            if type(opt) == "function" then
                local map = opt()
                for k, v in pairs(map) do
                    if v ~= nil then
                        option:dep(k):enable(v)
                    end
                end
            end
        end
        local bin_dir = option:dep("lc_bin_dir"):enabled()
        if is_mode("debug") then
            bin_dir = path.join(bin_dir, "debug")
        elseif is_mode("releasedbg") then
            bin_dir = path.join(bin_dir, "releasedbg")
        else
            bin_dir = path.join(bin_dir, "release")
        end
        option:set_value(bin_dir)
    end
    local lc_enable_tests = option:dep("lc_enable_tests")
    if lc_enable_tests:enabled() then
        option:dep("lc_enable_dsl"):enable(true, {
            force = true
        })
    end
    -- checking python
    local enable_py = option:dep("_lc_enable_py")
    local function non_empty_str(s)
        return type(s) == "string" and s:len() > 0
    end
    if not option:dep("lc_enable_py"):enabled() then
        enable_py:enable(false)
    elseif non_empty_str(option:dep("lc_py_include"):enabled()) then
        enable_py:enable(true)
    end
    local is_win = is_plat("windows")
    -- checking fallback
    local lc_fallback_backend = option:dep("lc_fallback_backend")
    local lc_llvm_path = option:dep("lc_llvm_path")
    local lc_embree_path = option:dep("lc_embree_path")
    local lc_enable_xir = option:dep("lc_enable_xir")
    if lc_fallback_backend:enabled() or lc_llvm_path:enabled() then
        lc_enable_xir:enable(true, {
            force = true
        })
    end
    if (not lc_llvm_path:enabled()) or (not lc_embree_path:enabled()) then
        lc_fallback_backend:enable(false, {
            force = true
        })
        if lc_fallback_backend:enabled() then
            utils.error("XIR not enabled. Fallback-backend force disabled.")
        end
    end

    -- checking dx
    local lc_dx_backend = option:dep("lc_dx_backend")
    if lc_dx_backend:enabled() and not is_win then
        lc_dx_backend:enable(false, {
            force = true
        })
        if lc_dx_backend:enabled() then
            utils.error("DX backend not supported in this platform, force disabled.")
        end
    end
    -- checking metal
    local lc_metal_backend = option:dep("lc_metal_backend")
    if lc_metal_backend:enabled() and not is_plat("macosx") then
        lc_metal_backend:enable(false, {
            force = true
        })
        if lc_metal_backend:enabled() then
            utils.error("Metal backend not supported in this platform, force disabled.")
        end
    end
    -- checking cuda
    local lc_cuda_ext_lcub = option:dep("lc_cuda_ext_lcub")
    local lc_cuda_backend = option:dep("lc_cuda_backend")
    if lc_cuda_backend:enabled() and not (is_win or is_plat("linux")) then
        lc_cuda_backend:enable(false, {
            force = true
        })
        if lc_cuda_backend:enabled() then
            utils.error("CUDA backend not supported in this platform, force disabled.")
        end
    end
    if lc_cuda_ext_lcub:enabled() and not lc_cuda_backend:enabled() then
        lc_cuda_ext_lcub:enable(false, {
            force = true
        })
        if lc_cuda_ext_lcub:enabled() then
            utils.error("CUDA lcub extension not supported when cuda is disabled")
        end
    end
    if enable_py:enabled() then
        option:dep("lc_enable_gui"):enable(true, {
            force = true
        })
    end
    -- dx cuda interop
    local lc_dx_cuda_interop = option:dep("lc_dx_cuda_interop")
    if lc_cuda_backend:enabled() and lc_dx_backend:enabled() then
        lc_dx_cuda_interop:enable(true)
    end
    -- vk cuda interop
    local lc_vk_cuda_interop = option:dep("lc_vk_cuda_interop")
    if lc_cuda_backend:enabled() and option:dep("lc_vk_backend"):enabled() then
        lc_vk_cuda_interop:enable(true)
    end
end)
option_end()
rule("lc_basic_settings")
on_config(function(target)
    if target:is_plat("linux") then
        -- Linux should use -stdlib=libc++
        -- https://github.com/LuisaGroup/LuisaCompute/issues/58
        if target:has_tool("cxx", "clang", "clangxx") then
            target:add("cxflags", "-stdlib=libc++", {
                force = true
            })
            target:add("syslinks", "c++")
        end
    end
    -- disable LTO
    -- if cc == "cl" then
    --     target:add("cxflags", "-GL")
    -- elseif cc == "clang" or cc == "clangxx" then
    --     target:add("cxflags", "-flto=thin")
    -- elseif cc == "gcc" or cc == "gxx" then
    --     target:add("cxflags", "-flto")
    -- end
    -- local _, ld = target:tool("ld")
    -- if ld == "link" then
    --     target:add("ldflags", "-LTCG")
    --     target:add("shflags", "-LTCG")
    -- elseif ld == "clang" or ld == "clangxx" then
    --     target:add("ldflags", "-flto=thin")
    --     target:add("shflags", "-flto=thin")
    -- elseif ld == "gcc" or ld == "gxx" then
    --     target:add("ldflags", "-flto")
    --     target:add("shflags", "-flto")
    -- end
end)
on_load(function(target)
    local _get_or = function(name, default_value)
        local v = target:extraconf("rules", "lc_basic_settings", name)
        if v == nil then
            v = target:values('lc_' .. name)
        end
        if v == nil then
            return default_value
        end
        return v
    end
    local toolchain = _get_or("toolchain", get_config("lc_toolchain"))
    if toolchain then
        target:set("toolchains", toolchain)
    end
    local project_kind = _get_or("project_kind", nil)
    if project_kind then
        target:set("kind", project_kind)
    end
    if target:is_plat("linux") then
        if project_kind == "static" or project_kind == "object" then
            target:add("cxflags", "-fPIC")
        end
    end
    if target:is_plat("macosx") then
        target:add("cxflags", "-no-pie")
        target:add("cxflags", "-Wno-invalid-specialization", {
            tools = {"clang"}
        })
    end
    -- fma support
    if target:is_arch("x64", "x86_64") then
        target:add("cxflags", "-mfma", {
            tools = {"clang", "gcc"}
        })
    end
    local c_standard = _get_or("c_standard", nil)
    local cxx_standard = _get_or("cxx_standard", nil)
    if type(c_standard) == "string" and type(cxx_standard) == "string" then
        target:set("languages", c_standard, cxx_standard, {
            public = true
        })
    else
        target:set("languages", "clatest", "cxx20", {
            public = true
        })
    end

    local enable_exception = _get_or("enable_exception", nil)
    if enable_exception then
        target:set("exceptions", "cxx")
    else
        target:set("exceptions", "no-cxx")
    end

    local win_runtime = get_config("lc_win_runtime")
    if is_mode("debug") then
        if not win_runtime then
            win_runtime = "MDd"
        end
        target:set("optimize", _get_or("optimize", "none"))
        target:add("cxflags", "/GS", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    elseif is_mode("releasedbg") then
        if not win_runtime then
            win_runtime = "MD"
        end
        target:set("optimize", _get_or("optimize", "none"))
        target:add("cxflags", "/GS-", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    else
        if not win_runtime then
            win_runtime = "MD"
        end
        target:set("optimize", _get_or("optimize", "aggressive"))
        target:add("cxflags", "/GS-", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    end
    target:set("warnings", _get_or("warnings", "none"))
    target:set("runtimes", _get_or("runtime", win_runtime), {
        public = true
    })
    target:set("fpmodels", _get_or("fpmodels", "fast"))
    target:add("cxflags", "/Zc:preprocessor", {
        tools = "cl",
        public = true
    });
    if _get_or("use_simd", has_config("lc_enable_simd")) then
        if is_arch("arm64") then
            target:add("vectorexts", "neon", {
                public = true
            })
        else
            target:add("vectorexts", "avx", "avx2", {
                public = true
            })
        end
    end
    local use_rtti = _get_or("rtti", false)
    if _get_or("no_rtti", not (use_rtti or has_config("lc_use_rtti") or has_config("_lc_enable_py"))) then
        target:add("cxflags", "/GR-", {
            tools = {"clang_cl", "cl"}
        })
        target:add("cxflags", "-fno-rtti", "-fno-rtti-data", {
            tools = {"clang"}
        })
        target:add("cxflags", "-fno-rtti", {
            tools = {"gcc"}
        })
    else
        target:add("cxflags", "/GR", {
            tools = {"clang_cl", "cl"}
        })
    end
end)
rule_end()

rule("lc-rename-ext")
on_load(function(target)
    target:set("basename", "luisa-ext-" .. target:name())
end)
rule_end()

target("lc-check-winsdk")
set_kind("phony")
on_config(function(target)
    if not target:is_plat("windows") then
        return
    end
    local toolchain_settings = target:toolchain("msvc")
    if not toolchain_settings then
        toolchain_settings = target:toolchain("clang-cl")
    end
    if not toolchain_settings then
        toolchain_settings = target:toolchain("llvm")
    end
    if not toolchain_settings then
        return
    end
    local sdk_version = toolchain_settings:runenvs().WindowsSDKVersion
    local legal_sdk = false
    if sdk_version then
        import("core.base.semver")
        local ver = semver.match(sdk_version)
        if ver then
            if ver:major() > 10 then
                legal_sdk = true
            elseif ver:major() == 10 then
                if ver:patch() >= 22000 then
                    legal_sdk = true
                end
            end
        end
        if not legal_sdk then
            raise("Illegal windows SDK version, requires 10.0.22000.0 or later")
        end
    end
end)
target_end()
rule("build_cargo")
set_extensions(".toml")
on_buildcmd_file(function(target, batchcmds, sourcefile, opt)
    local lib = import("lib")
    local sb = lib.StringBuilder("cargo build -q ")
    -- if backend_off then
    sb:add("--no-default-features ")
    -- end
    sb:add("--manifest-path ")
    sb:add(sourcefile):add(' ')
    local features = target:get('features')
    if features then
        sb:add("--features ")
        sb:add(features):add(' ')
    end
    if not is_mode("debug") then
        sb:add("--release ")
    end
    local cargo_cmd = sb:to_string()
    batchcmds:show(cargo_cmd)
    batchcmds:vrun(cargo_cmd)
    sb:dispose()
end)
rule_end()

-- Support:

-- add_rules('lc_install_sdk', {
--     sdk_dir = xxx
--     libnames = {{
--         name = yyy,
--         extract_dir = xxx                 -- extract to default dir or target dir
--         copy_dir = ""                    -- no copy or copy to target dir
--         plat_spec = true                 -- name will be transformed to yyyy-linux-x64.zip
--     }}
-- })

-- add_rules('lc_install_sdk', {
--     libnames = xxx
--     
-- })

-- and

-- add_rules('lc_install_sdk', {
--     sdk_dir = xxx
--     libnames = {
--         name = yyy
--     }
-- })
rule('lc_install_sdk')
-- if compile depend on installing, use:
-- set_policy('build.fence', true)
before_build(function(target)
    import("find_sdk")
    find_sdk.on_install_sdk(target, 'lc_install_sdk')
end)
rule_end()

rule("lc_compile_codegen")
set_extensions(".lua")
on_build_files(function(target, jobgraph, sourcebatch, opt)
    local var_name_prefix = target:extraconf("rules", "lc_compile_codegen", "var_name_prefix")
    if not var_name_prefix then
        var_name_prefix = "_"
    end
    local remove_ext = target:extraconf("rules", "lc_compile_codegen", "remove_ext")
    local remove_slash_r = target:extraconf("rules", "lc_compile_codegen", "remove_slash_r")
    for _, sourcefile in ipairs(sourcebatch.sourcefiles) do
        local filename = path.filename(sourcefile)
        local rootdir = path.directory(sourcefile)
        local header_lib = import(path.basename(filename), {
            rootdir = rootdir
        })
        local src_dir = header_lib.src_dir()

        local process_job = sourcefile .. "/process"

        if src_dir then
            jobgraph:add(process_job, function()
                local codegen_dir = path.join(target:targetdir(), "lc_embed_codegen")
                local args = {}
                table.insert(args, src_dir)
                table.insert(args, header_lib.dst_file())
                table.insert(args, header_lib.meta_dir())
                table.insert(args, var_name_prefix)
                if remove_ext then
                    table.insert(args, "y")
                else
                    table.insert(args, "n")
                end
                if remove_slash_r then
                    table.insert(args, "y")
                else
                    table.insert(args, "n")
                end
                local files = header_lib.file_list()
                for _, v in ipairs(files) do
                    table.insert(args, v)
                end
                os.runv(codegen_dir, args)
            end)
            jobgraph:group(sourcefile, function()
                local batchcxx = {
                    rulename = "c++.build",
                    sourcekind = "cxx",
                    sourcefiles = {},
                    objectfiles = {},
                    dependfiles = {}
                }
                local dst_name = header_lib.dst_file()
                local objectfile = target:objectfile(dst_name)
                local dependfile = target:dependfile(objectfile)
                table.insert(target:objectfiles(), objectfile)
                table.insert(batchcxx.objectfiles, objectfile)
                table.insert(batchcxx.dependfiles, dependfile)
                table.insert(batchcxx.sourcefiles, dst_name)
                import("private.action.build.object")(target, jobgraph, batchcxx, opt)
            end)
            jobgraph:add_orders(process_job, sourcefile)
        end
    end
end, {
    jobgraph = true
})
rule_end()

rule('lc_llvm')
on_load(function(target, opt)
    local libs = {}
    local lc_llvm_path = get_config("lc_llvm_path")
    if not lc_llvm_path then
        return
    end
    target:add("linkdirs", path.join(lc_llvm_path, "lib"))
    target:add("includedirs", path.join(lc_llvm_path, "include"))
    for __, filepath in ipairs(os.files(path.join(lc_llvm_path, "lib/*.lib"))) do
        local basename = path.basename(filepath)
        if basename:match("LLVM") ~= nil and basename ~= "LLVM-C" then
            table.insert(libs, basename)
        end
    end
    target:add("links", libs)
    if is_plat("windows") then
        target:add("syslinks", "Version", "advapi32", "Shcore", "user32", "shell32", "Ole32", 'Ws2_32', 'ntdll')
    elseif is_plat("linux") then
        target:add("syslinks", "uuid")
    elseif is_plat("macosx") then
        target:add("frameworks", "CoreFoundation")
    end
end)
after_build(function(target)
    import("async.jobgraph")
    import("async.runjobs")
    if not is_plat("windows") then
        return
    end
    local lc_llvm_path = get_config("lc_llvm_path")
    if not lc_llvm_path then
        return
    end
    local function copy(src_path, dst_path)
        os.cp(src_path, dst_path, {
            copy_if_different = true,
            async = true,
            detach = true
        })
    end
    local dst_path = target:targetdir()
    local jobs = jobgraph.new()
    for __, filepath in ipairs(os.files(path.join(lc_llvm_path, "bin/*.dll"))) do
        jobs.add(filepath, function()
            copy(filepath, path.join(dst_path, path.filename(filepath)))
        end)
    end
    runjobs("copy_llvm_job", jobs, {
        comax = 1000,
        timeout = -1,
        timer = function(running_jobs_indices)
            utils.error("timeout.")
        end
    })
end)
rule_end()

rule("lc_run_target")
on_run(function(target)
    import("core.base.option")
    local name = target:extraconf("rules", "lc_run_target", "name")
    if not name then
        name = target:name()
    end
    local arguments = option.get("arguments")
    local tar_dir = path.absolute(target:targetdir())
    os.execv(path.join(tar_dir, name), arguments, {
        curdir = tar_dir
    })
end)
rule_end()

-- In-case of submod, when there is override rules, do not overload
if _config_rules == nil then
    _config_rules = {"lc_basic_settings"}
end
if _disable_unity_build == nil then
    local unity_build = get_config("lc_enable_unity_build")
    if unity_build ~= nil then
        _disable_unity_build = not unity_build
    end
end
if not _config_project then
    function _config_project(config)
        local batch_size = config["batch_size"]
        if type(batch_size) == "number" and batch_size > 1 and (not _disable_unity_build) then
            add_rules("c.unity_build", {
                batchsize = batch_size
            })
            add_rules("c++.unity_build", {
                batchsize = batch_size
            })
        end
        if type(_config_rules) == "table" then
            add_rules(_config_rules, config)
        end
    end
end
