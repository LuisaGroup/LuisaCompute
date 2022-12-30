_config_project({
	project_name = "lc-ir",
	project_kind = "shared"
})
local add_includedirs = _get_add_includedirs()
add_defines("LC_IR_EXPORT_DLL")
add_deps("lc-runtime", "lc-rust")
add_files("**.cpp")
add_includedirs("../rust", {
	public = true
})
if is_plat("windows") then
	add_syslinks("Ws2_32", "Advapi32", "Bcrypt", "Userenv")
end
function add_rs_link(str)
	add_links("src/rust/target/" .. str .. "/luisa_compute_api_types", "src/rust/target/" .. str .. "/luisa_compute_ir")
end
if is_mode("debug") then
	add_rs_link("debug")
else
	add_rs_link("release")
end