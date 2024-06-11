target("lmdb")
_config_project({
    project_kind = "object"
})
add_files("mdb.c", "midl.c")
add_includedirs("./", {public = true})
if is_plat("windows") then
    add_syslinks("Advapi32", {public = true})    
end
target_end()