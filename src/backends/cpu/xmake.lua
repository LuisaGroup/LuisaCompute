target("lc-backend-cpu")
_config_project({
	project_kind = "shared"
})
on_load(function(target)
	local function rela(p)
		return path.relative(path.absolute(p, os.scriptdir()), os.projectdir())
	end
	target:add("defines", "LC_IR_EXPORT_DLL")
	target:add("deps", "lc-runtime", "lc-ir", "lc-ast", "lc-rust")
	target:set("features", "cpu")
	if is_plat("windows") then
		target:add("syslinks", "Ws2_32", "Advapi32", "Bcrypt", "Userenv")
	end

	local function add_rs_link(str)
		local lib_path = path.absolute(path.join("../../rust/target", str), os.scriptdir())
		target:add("linkdirs", path.absolute(path.join("../../rust/target", str), os.scriptdir()), {
			public = true
		})
		if is_plat("windows") then
			target:add("links", "luisa_compute_backend_impl.dll", {
				public = true
			})
		elseif is_plat("linux") then
			target:add("links", path.join(lib_path, "libluisa_compute_backend_impl.so"), {
				public = true
			})
		else
			target:add("links", path.join(lib_path, "libluisa_compute_backend_impl.dylib"), {
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
before_build(function(target)
	local profile = nil
	if is_mode("debug") then
		profile = "debug"
	else
		profile = "release"
	end
	local lib_path = path.absolute(path.join("../../rust/target", profile), os.scriptdir())
	os.setenv("EMBREE_DLL_OUT_DIR", lib_path)
end)
add_rules("build_cargo")
add_files("../../rust/luisa_compute_backend_impl/Cargo.toml")

after_build(function(target)
	local dlls = {
		"embree4",
		"tbbmalloc",
		"luisa_compute_backend_impl"
	}
	if is_plat("windows") then
		table.insert(dlls, "libmmd")
		table.insert(dlls, "tbb12")
	else
		table.insert(dlls, "tbb")
	end
	local function copy_dll(str)
		local lib_path = path.absolute(path.join("../../rust/target", str), os.scriptdir())
		local bin_dir = target:targetdir()
		for i, v in ipairs(dlls) do
			if is_plat("windows") then
				os.cp(path.join(lib_path, v .. ".dll"), bin_dir)
			elseif is_plat("linux") then
				os.cp(path.join(lib_path, 'lib' .. v .. ".so"), bin_dir)
			else
				-- macOS compiles from source, so ignore the copy error if any
				local dylib = path.join(lib_path, 'lib' .. v .. ".dylib")
				if os.isfile(dylib) then
					os.cp(dylib, bin_dir)
				end
			end
		end
	end
	if is_mode("debug") then
		copy_dll("debug")
	else
		copy_dll("release")
	end
end)
add_files("**.cpp", "../common/rust_device_common.cpp")
target_end()
