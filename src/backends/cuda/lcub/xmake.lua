rule("cuda_ext")
on_load(function(target)
	local cuda_path = os.getenv("CUDA_PATH")
	if cuda_path then
		target:add("sysincludedirs", path.join(cuda_path, "include"), {public=true})
		target:add("linkdirs", path.join(cuda_path, "lib/x64/"), {public=true})
		target:add("links", "nvrtc", "cudart", "cuda", {public=true})
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

target("lc-backend-cuda-ext-dcub")
    add_rules("cuda_ext")
    set_toolchains("cuda") -- compiler: nvcc
    set_languages("cxx17")
    set_kind("shared")
    add_files("private/dcub/*.cu")
	add_includedirs("$(projectdir)/include")
    add_cuflags("-DDCUB_DLL_EXPORTS",{public=false})
    add_cugencodes("native")
    add_cugencodes("compute_75")
target_end()


target("lc-backend-cuda-ext-lcub")
    set_languages("cxx20")
    set_kind("shared")
	add_deps("lc-backend-cuda-ext-dcub")
	add_deps("lc-runtime")
    add_files("*.cpp")
target_end()
