target("lc-vstl")
_config_project({
    project_kind = "shared",
    batch_size = 4
})
add_deps("lc-core")
set_pcxxheader("pch.h")
add_headerfiles("../../include/luisa/vstl/**.h")
add_files("**.cpp")
add_defines("LC_VSTL_EXPORT_DLL")
if is_plat("windows") then
    add_syslinks("Ole32")
elseif is_plat("linux") then
    add_syslinks("uuid")
elseif is_plat("macosx") then
    add_frameworks("CoreFoundation")
end
target_end()
