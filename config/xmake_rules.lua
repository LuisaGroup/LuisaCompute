-- Add project's include directories
rule("add_lc_includedirs")
on_load(function(target)
	local is_public = target:values("lc_is_public")
	if is_public == nil then
		utils.error("'lc_is_public' must be set before call add_lc_includedirs")
		return
	end
	local lc_dir = target:values("lc_dir") or false
	if not lc_dir then
		utils.error("'lc_dir' must be set before call add_lc_includedirs")
		return
	end
	target:add("includedirs", 
		--[[pybind11]]
		path.join(lc_dir,"src/ext/pybind11/include"),
		--[[spdlog]]
		path.join(lc_dir,"src/ext/spdlog/include"),
		--[[half]]
		path.join(lc_dir,"src/ext/half/include"),
		--[[mimalloc]]
		path.join(lc_dir,"src/ext/EASTL/packages/mimalloc/include"),
		--[[eastl]]
		path.join(lc_dir,"src/ext/EASTL/include"), path.join(lc_dir,"src/ext/EASTL/packages/EABase/include/Common"),
		--[[lc-core]]
		path.join(lc_dir,"include"), path.join(lc_dir,"src/ext/xxHash"), path.join(lc_dir,"src/ext/magic_enum/include"),
		--[[imgui]]
		path.join(lc_dir, "src/ext/imgui"), path.join(lc_dir, "src/ext/imgui/backends"),
		{
			public = is_public
		}
	)
	local marl_include = path.join(lc_dir,"src/ext/marl/include")
	if os.exists(marl_include) then
		target:add("includedirs", marl_include, {
			public = is_public
		})
	end
	if is_plat("windows") then
		target:add("syslinks", "Ole32", "Advapi32", {
			public = is_public
		})
	elseif is_plat("linux") then
		target:add("syslinks", "uuid", {
			public = is_public
		})
	else
		target:add("frameworks", "CoreFoundation", {
			public = is_public
		})
	end
end)
rule_end()

rule("add_lc_defines")
on_load(function(target)
	local is_public = target:values("lc_is_public") or false
	if is_public == nil then
		utils.error("'lc_is_public' must be set before call add_lc_defines")
		return
	end
	target:add("defines", 
	--[[spdlog]]
		"SPDLOG_NO_EXCEPTIONS", "SPDLOG_NO_THREAD_ID", "SPDLOG_DISABLE_DEFAULT_LOGGER",
		"FMT_SHARED", "FMT_CONSTEVAL=constexpr", "FMT_USE_CONSTEXPR=1",
		"FMT_EXCEPTIONS=0", 
	--[[eastl]] 
		"EA_PRAGMA_ONCE_SUPPORTED",
		"EASTL_ASSERT_ENABLED=0", "EA_HAVE_CPP11_CONTAINERS", "EA_HAVE_CPP11_ATOMIC", "EA_HAVE_CPP11_CONDITION_VARIABLE",
		"EA_HAVE_CPP11_MUTEX", "EA_HAVE_CPP11_THREAD", "EA_HAVE_CPP11_FUTURE", "EA_HAVE_CPP11_TYPE_TRAITS",
		"EA_HAVE_CPP11_TUPLES", "EA_HAVE_CPP11_REGEX", "EA_HAVE_CPP11_RANDOM", "EA_HAVE_CPP11_CHRONO",
		"EA_HAVE_CPP11_SCOPED_ALLOCATOR", "EA_HAVE_CPP11_INITIALIZER_LIST", "EA_HAVE_CPP11_SYSTEM_ERROR",
		"EA_HAVE_CPP11_TYPEINDEX", "EASTL_STD_ITERATOR_CATEGORY_ENABLED", "EASTL_STD_TYPE_TRAITS_AVAILABLE",
		"EASTL_MOVE_SEMANTICS_ENABLED", "EASTL_VARIADIC_TEMPLATES_ENABLED", "EASTL_VARIABLE_TEMPLATES_ENABLED",
		"EASTL_INLINE_VARIABLE_ENABLED", "EASTL_HAVE_CPP11_TYPE_TRAITS", "EASTL_INLINE_NAMESPACES_ENABLED",
		"EASTL_ALLOCATOR_EXPLICIT_ENABLED", "EASTL_USER_DEFINED_ALLOCATOR", "EA_DLL",
		"LUISA_VSTL_STATIC_LIB", "LUISA_DSL_STATIC_LIB",
	--[[imgui]]
		"ImDrawIdx=unsigned int", "IMGUI_API=__declspec(dllimport)",
	--[[marl]]
		"MARL_DLL",
	{
		public = is_public
	})
	if is_plat("windows") then
		target:add("defines", "IMGUI_API=__declspec(dllimport)", {public = is_public})
	end
end)
rule_end()
