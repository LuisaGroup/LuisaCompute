target("lc-ir")
_config_project({
	project_kind = "shared"
})
on_load(function(target)
	local function rela(p)
		return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
	end
	target:add("defines", "LC_IR_EXPORT_DLL")
	target:add("deps", "lc-ast", "lc-rust")
	if is_plat("windows") then
		target:add("syslinks", "Ws2_32", "Advapi32", "Bcrypt", "Userenv")
	end
	local function add_rs_link(str)
		local lib_path = path.absolute(path.join("../rust/target", str), os.scriptdir())
		target:add("linkdirs", lib_path, {
			public = true
		})
		if is_plat("windows") then
			target:add("links", "luisa_compute_ir_static", {
				public = true
			})
			-- missing libraries can be found using `cargo rustc -- --print=native-static-libs`
			target:add("links", "ntdll", {
				public = true
			})
		elseif is_plat("linux") then
			target:add("links", path.join(lib_path, "libluisa_compute_ir_static.a"), {
				public = true
			})
			target:add("rpathdirs", lib_path)
		else
			target:add("links", path.join(lib_path, "libluisa_compute_ir_static"), {
				public = true
			})
		end
	end
	if is_mode("debug") then
		add_rs_link("debug")
	else
		add_rs_link("release")
	end
end)
add_headerfiles("../../include/luisa/ir/**.h")
add_files("**.cpp")
target_end()
