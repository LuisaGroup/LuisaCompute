target("lc-runtime")
_config_project({
	project_kind = "shared",
	batch_size = 8
})
add_deps("lc-ast")
set_pcxxheader("pch.h")
add_defines("LC_RUNTIME_EXPORT_DLL")
if get_config("enable_ir") then
	add_defines("LUISA_ENABLE_IR", {
		public = true
	})
	add_deps("lc-ir")
end
add_headerfiles("../../include/luisa/runtime/**.h")
add_files("**.cpp")
target_end()
