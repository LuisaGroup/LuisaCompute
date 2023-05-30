target("lc-ir")
_config_project({
	project_kind = "shared"
})
on_load(function(target)
	local function rela(p)
		return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
	end
	target:add("defines", "LC_IR_EXPORT_DLL")
	target:add("deps", "lc-runtime", "lc-rust")
	target:add("includedirs", rela("../rust"), {
		public = true
	})
	if is_plat("windows") then
		target:add("syslinks", "Ws2_32", "Advapi32", "Bcrypt", "Userenv")
	end
	local function add_rs_link(str)
		target:add("linkdirs", path.absolute(path.join("../rust/target", str), os.scriptdir()))
		target:add("links", "luisa_compute_api_types", "luisa_compute_ir.dll")
	end
	if is_mode("debug") then
		add_rs_link("debug")
	else
		add_rs_link("release")
	end
end)
add_files("**.cpp")
target_end()
