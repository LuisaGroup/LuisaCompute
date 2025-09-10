option("_lc_enable_py")
set_showmenu(false)
set_default(false)
option_end()

option("_lc_enable_rust")
set_showmenu(false)
set_default(false)
option_end()

option("_lc_vk_sdk_dir")
set_default(false)
set_showmenu(false)
add_deps("vk_support")
after_check(function(option)
    if not option:dep("vk_support"):enabled() then
        option:set_value(false)
        return
    end
    local sdk_dir = os.getenv("VK_SDK_PATH")
    if not sdk_dir then
        sdk_dir = os.getenv("VULKAN_SDK")
    end
    if not sdk_dir then
        option:set_value(false)
    else
        option:set_value(sdk_dir)
    end
end)
option_end()

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
add_deps("dx_backend", "vk_backend", "cuda_backend",
    "metal_backend", "cpu_backend", "enable_tests", "py_include",
    "cuda_ext_lcub", "enable_ir", "enable_dsl", "enable_clangcxx",
    "enable_gui", "bin_dir", "_lc_enable_py", "_lc_enable_rust")
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
        local bin_dir = option:dep("bin_dir"):enabled()
        if is_mode("debug") then
            bin_dir = path.join(bin_dir, "debug")
        elseif is_mode("releasedbg") then
            bin_dir = path.join(bin_dir, "releasedbg")
        else
            bin_dir = path.join(bin_dir, "release")
        end
        option:set_value(bin_dir)
    end
    local enable_tests = option:dep("enable_tests")
    if enable_tests:enabled() then
        option:dep("enable_dsl"):enable(true, {
            force = true
        })
    end
    -- checking python
    local enable_py = option:dep("_lc_enable_py")
    local function non_empty_str(s)
        return type(s) == "string" and s:len() > 0
    end
    if non_empty_str(option:dep("py_include"):enabled()) then
        enable_py:enable(true)
    end
    local is_win = is_plat("windows")
    -- checking dx
    local dx_backend = option:dep("dx_backend")
    if dx_backend:enabled() and not is_win then
        dx_backend:enable(false, {
            force = true
        })
        if dx_backend:enabled() then
            utils.error("DX backend not supported in this platform, force disabled.")
        end
    end
    -- checking metal
    local metal_backend = option:dep("metal_backend")
    if metal_backend:enabled() and not is_plat("macosx") then
        metal_backend:enable(false, {
            force = true
        })
        if metal_backend:enabled() then
            utils.error("Metal backend not supported in this platform, force disabled.")
        end
    end
    -- checking cuda
    local cuda_ext_lcub = option:dep("cuda_ext_lcub")
    local cuda_backend = option:dep("cuda_backend")
    if cuda_backend:enabled() and not (is_win or is_plat("linux")) then
        cuda_backend:enable(false, {
            force = true
        })
        if cuda_backend:enabled() then
            utils.error("CUDA backend not supported in this platform, force disabled.")
        end
    end
    if cuda_ext_lcub:enabled() and not cuda_backend:enabled() then
        cuda_ext_lcub:enable(false, {
            force = true
        })
        if cuda_ext_lcub:enabled() then
            utils.error("CUDA lcub extension not supported when cuda is disabled")
        end
    end
    if enable_py:enabled() then
        option:dep("enable_gui"):enable(true, {
            force = true
        })
    end
    -- checking rust
    local enable_ir = option:dep("enable_ir")
    local cpu_backend = option:dep("cpu_backend")
    if not enable_ir:enabled() then
        option:dep("_lc_enable_rust"):set_value(false)
        cpu_backend:enable(false, {
            force = true
        })
    else
        import("lib.detect.find_tool")
        local rust_cargo = find_tool("cargo") ~= nil
        option:dep("_lc_enable_rust"):set_value(rust_cargo)
        if not rust_cargo then
            enable_ir:enable(false)
            cpu_backend:enable(false)
            if enable_ir:enabled() then
                utils.error("Cargo not installed, IR module force disabled.")
                enable_ir:enable(false, {
                    force = true
                })
            end
            if cpu_backend:enabled() then
                utils.error("Cargo not installed, CPU backend force disabled.")
                cpu_backend:enable(false, {
                    force = true
                })
            end
        end
    end
end)
option_end()
rule("lc_basic_settings")
on_config(function(target)
    local _, cc = target:tool("cxx")
    if is_plat("linux") then
        -- Linux should use -stdlib=libc++
        -- https://github.com/LuisaGroup/LuisaCompute/issues/58
        if (cc == "clang" or cc == "clangxx") then
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
    if is_plat("linux") then
        if project_kind == "static" or project_kind == "object" then
            target:add("cxflags", "-fPIC", {
                tools = {"clang", "gcc"}
            })
        end
    end
    if is_plat("macosx") then
        target:add("cxflags", "-no-pie")
        target:add("cxflags", "-Wno-invalid-specialization", {
            tools = {"clang"}
        })
    end
    -- fma support
    if is_arch("x64", "x86_64") then
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

    local force_optimize = _get_or("force_optimize", nil)
    local win_runtime = get_config("lc_win_runtime")
    if is_mode("debug") then
        if not win_runtime then
            win_runtime = "MDd"
        end
        if force_optimize then
            target:set("optimize", "aggressive")
        else
            target:set("optimize", "none")
        end
        target:add("cxflags", "/GS", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    elseif is_mode("releasedbg") then
        if not win_runtime then
            win_runtime = "MDd"
        end
        if force_optimize then
            target:set("optimize", "aggressive")
        else
            target:set("optimize", "none")
        end
        target:add("cxflags", "/GS-", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    else
        if not win_runtime then
            win_runtime = "MD"
        end
        target:set("optimize", "aggressive")
        target:add("cxflags", "/GS-", "/Gd", {
            tools = {"clang_cl", "cl"},
            public = true
        })
    end
    target:set("warnings", "none")
    target:set("runtimes", _get_or("runtime", win_runtime), {
        public = true
    })
    target:set("fpmodels", "fast")
    target:add("cxflags", "/Zc:preprocessor", {
        tools = "cl",
        public = true
    });
    if _get_or("use_simd", get_config("enable_simd")) then
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
    if _get_or("no_rtti",  not (use_rtti or get_config("lc_use_rtti") or get_config("_lc_enable_py"))) then
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
    target:set("basename", "lc-ext-" .. target:name())
end)
rule_end()

target("lc-check-winsdk")
set_kind("phony")
on_config(function(target)
    if not is_plat("windows") then
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
        local lib = import("lib")
        local vers = lib.string_split(sdk_version, '.')
        if #vers > 0 then
            if tonumber(vers[1]) > 10 then
                legal_sdk = true
            elseif tonumber(vers[1]) == 10 then
                if #vers > 2 then
                    if tonumber(vers[3]) >= 22000 then
                        legal_sdk = true
                    end
                end
            end
        end
        if not legal_sdk then
            os.raise("Illegal windows SDK version, requires 10.0.22000.0 or later")
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
    batchcmds:vrunv(cargo_cmd)
    sb:dispose()
end)
rule_end()

rule('lc_install_sdk')
on_load(function(target)
    local custom_sdk_dir = get_config("sdk_dir")
    if type(custom_sdk_dir ) == "string" and not os.exists(custom_sdk_dir) then
        return
    end
    local packages = import('packages')
    local libnames = target:extraconf("rules", "lc_install_sdk", "libnames")
    local find_sdk = import('find_sdk')
    local enable = true
    for _, lib in ipairs(libnames) do
        local valid = find_sdk.check_file(lib, custom_sdk_dir)
        if not valid then
            utils.error("Library: " .. packages.sdks()[lib]['name'] ..
                            " not installed, run 'xmake lua setup.lua' or download it manually from " ..
                            packages.sdk_address(packages.sdks()[lib]) .. ' to ' .. packages.sdk_dir(os.arch(), custom_sdk_dir) ..
                            '.')
            enable = false
        end
    end
    if not enable then
        target:set('enabled', false)
    end
end)
on_clean(function(target)
    local custom_sdk_dir = get_config("sdk_dir")
    if type(custom_sdk_dir ) == "string" and not os.exists(custom_sdk_dir) then
        return
    end
    local bin_dir = target:targetdir()
    local find_sdk = import('find_sdk')
    local packages = import('packages')
    local sdks = packages.sdks()
    local libnames = target:extraconf("rules", "lc_install_sdk", "libnames")
    for _, lib in ipairs(libnames) do
        local sdk_map = sdks[lib]
        local cache_file_name = path.join(bin_dir, lib .. '.txt')
        if os.exists(cache_file_name) then
            os.rm(cache_file_name)
        end
    end
end)
before_build(function(target)
    local custom_sdk_dir = get_config("sdk_dir")
    if type(custom_sdk_dir ) == "string" and not os.exists(custom_sdk_dir) then
        return
    end
    local bin_dir = target:targetdir()
    local lib = import('lib')
    lib.mkdirs(bin_dir)
    local libnames = target:extraconf("rules", "lc_install_sdk", "libnames")
    local packages = import('packages')
    local find_sdk = import('find_sdk')
    local sdks = packages.sdks()
    local sdk_dir = packages.sdk_dir(os.arch(), custom_sdk_dir)
    for _, lib in ipairs(libnames) do
        local sdk_map = sdks[lib]
        local zip = sdk_map['name']
        local cache_file_name = path.join(bin_dir, lib .. '.txt')
        local data
        if os.exists(cache_file_name) then
            data = io.readfile(cache_file_name)
        end
        if not data or data ~= sdk_map['sha256'] then
            io.writefile(cache_file_name, sdk_map['sha256'])
            find_sdk.unzip_sdk(zip, sdk_dir, bin_dir)
        end
    end
end)
rule_end()

-- In-case of submod, when there is override rules, do not overload
if _config_rules == nil then
    _config_rules = {"lc_basic_settings"}
end
if _disable_unity_build == nil then
    local unity_build = get_config("enable_unity_build")
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
