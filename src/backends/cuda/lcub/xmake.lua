rule("cuda_ext")
on_load(function(target)
    local cuda_path = os.getenv("CUDA_PATH")
    if cuda_path then
        target:add("sysincludedirs", path.join(cuda_path, "include"), {
            public = true
        })
        target:add("linkdirs", path.join(cuda_path, "lib/x64/"), {
            public = true
        })
        target:add("links", "nvrtc", "cudart", "cuda", {
            public = true
        })
    else
        target:set("enabled", false)
        return
    end
    if is_plat("windows") then
        target:add("defines", "NOMINMAX", "UNICODE")
        target:add("syslinks", "Cfgmgr32", "Advapi32")
    end
end)
rule_end()

target("lcub_env")
set_kind("phony")
on_load(function(target)
    if is_plat("windows") then
        import("detect.sdks.find_vstudio")
        local tool = find_vstudio()
        local max_version = 0
        local key = nil
        for version, dict in pairs(tool) do
            local ver_num = tonumber(version)
            if ver_num >= max_version then
                max_version = ver_num
                key = version
            end
            break
        end
        if not key then
            target:set_enabled(false)
            utils.error("Can not find Visual Studio. lcub disabled.")
            return
        end
        local vcvarsall = tool[key]["vcvarsall"]
        local vs_dict
        if vcvarsall then
            if is_arch("x64") then
                vs_dict = vcvarsall["x64"]
            elseif is_arch("arm64") then
                vs_dict = vcvarsall["arm64"]
            end
        end
        if not vs_dict then
            target:set_enabled(false)
            utils.error("Can not find Visual Studio. lcub disabled.")
            return
        end
        os.setenv("PATH", vs_dict["PATH"])
    end
end)
target_end()

target("luisa-compute-cuda-ext-dcub")
add_rules("cuda_ext")
set_toolchains("cuda") -- compiler: nvcc
set_languages("cxx17")
set_kind("shared")
add_files("private/dcub/*.cu")
add_includedirs("../../../../include")
add_cuflags("-DDCUB_DLL_EXPORTS", {
    public = false
})
add_cugencodes("native")
add_cugencodes("compute_75")
add_deps("lcub_env")
target_end()

target("luisa-compute-cuda-ext-lcub")
set_languages("cxx20")
set_kind("shared")
add_deps("luisa-compute-cuda-ext-dcub", "lc-runtime")
add_files("*.cpp")
target_end()
