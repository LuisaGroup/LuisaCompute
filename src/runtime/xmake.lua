target("lc-runtime")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-ast")
add_defines("LC_RUNTIME_EXPORT_DLL")
if get_config("_lc_enable_rust") then
	add_defines("LUISA_ENABLE_API", {
		public = true
	})
	add_defines("LUISA_ENABLE_IR", {
		public = true
	})
end
add_headerfiles("../../include/luisa/runtime/**.h")
add_files("**.cpp")
target_end()
