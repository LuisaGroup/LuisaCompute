target("lc-rust")
_config_project({
    project_kind = "object"
})
rule("build_cargo")
set_extensions(".toml")
on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
	-- local sub_dir = "src/rust/Cargo.toml"
	local sub_dir = sourcefile
	local cargo_cmd = "cargo build -q --manifest-path "
	local mode = nil
	if is_mode("debug") then
		cargo_cmd = cargo_cmd .. sub_dir
	else
		cargo_cmd = cargo_cmd .. sub_dir .. " --release"
	end
	print(cargo_cmd)
	-- batchcmds:vrunv(cargo_cmd)
	os.run(cargo_cmd)
end)
rule_end()
add_rules("build_cargo")
add_files("luisa_compute_ir/Cargo.toml")
add_files("luisa_compute_api_types/Cargo.toml")
add_files("luisa_compute_cpu_kernel_defs/Cargo.toml")