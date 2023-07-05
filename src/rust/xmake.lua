target("lc-rust")
_config_project({
	project_kind = "object"
})
add_rules("build_cargo")
add_files("Cargo.toml")
add_files("luisa_compute_ir/Cargo.toml")
-- add_files("luisa_compute_backend/Cargo.toml")
-- add_files("luisa_compute_backend_impl/Cargo.toml")
-- add_files("luisa_compute_cpu_kernel_defs/Cargo.toml")
target_end()
