target("lc-vstl")
_config_project({
    project_kind = "static",
    batch_size = 4
})
add_deps("lc-core", "lmdb")
set_pcxxheader("pch.h")
add_headerfiles("../../include/luisa/vstl/**.h")
add_defines("LUISA_VSTL_STATIC_LIB", {
    public = true
})
add_files("**.cpp")
if is_plat("windows") then
    add_syslinks("Ole32", {
        public = true
    })
elseif is_plat("linux") then
    add_syslinks("uuid", {
        public = true
    })
elseif is_plat("macosx") then
    add_frameworks("CoreFoundation", {
        public = true
    })
end
target_end()
