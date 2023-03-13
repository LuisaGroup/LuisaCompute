target("lc-rust")
_config_project({
    project_kind = "object"
})
rule("build_cargo")
set_extensions(".toml")
on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
	local sub_dir = sourcefile
	local cargo_cmd = "cargo build --manifest-path "
	local mode = nil
	if is_mode("debug") then
		cargo_cmd = cargo_cmd .. sub_dir
	else
		cargo_cmd = cargo_cmd .. sub_dir .. " --release"
	end
	print("run: " .. cargo_cmd)
	batchcmds:vrunv(cargo_cmd)
end)
rule_end()
add_rules("build_cargo")
add_files("Cargo.toml")