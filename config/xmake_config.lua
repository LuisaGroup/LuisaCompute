-- Add project's include directories
function add_lc_includedirs(lc_dir, is_public)
	add_includedirs(
	--[[spdlog]]
		path.join(lc_dir, "src/ext/spdlog/include"),
	--[[mimalloc]]
		path.join(lc_dir, "src/ext/EASTL/packages/mimalloc/include"),
	--[[eastl]]
		path.join(lc_dir, "src/ext/EASTL/include"), path.join(lc_dir, "src/ext/EASTL/packages/EABase/include/Common"),
	--[[lc-core]]
		path.join(lc_dir, "src"), path.join(lc_dir, "src/ext/xxHash"), path.join(lc_dir, "src/ext/parallel-hashmap"), 
		
	{
		public = is_public
	})
end
-- Add project's link dir
function add_lc_linkdirs(lc_dir, is_public)
	if is_mode("debug") then
		add_linkdirs(path.join(lc_dir, "bin/debug"), {
			public = true
		})
	else
		add_linkdirs(path.join(lc_dir, "bin/release"), {
			public = true
		})
	end
end
-- Add project's defines
function add_lc_defines(lc_dir, is_public)
	add_defines(
	--[[spdlog]]
		"SPDLOG_NO_EXCEPTIONS", "SPDLOG_NO_THREAD_ID", "SPDLOG_DISABLE_DEFAULT_LOGGER",
		"SPDLOG_COMPILED_LIB", "FMT_SHARED", "SPDLOG_SHARED_LIB", "FMT_CONSTEVAL=constexpr", "FMT_USE_CONSTEXPR=1",
		"FMT_EXCEPTIONS=0", 
	--[[mimallo]]
		"MI_SHARED_LIB",
	--[[eastl]] 
		"EA_PRAGMA_ONCE_SUPPORTED",
		"EASTL_ASSERT_ENABLED=0", "EA_HAVE_CPP11_CONTAINERS", "EA_HAVE_CPP11_ATOMIC", "EA_HAVE_CPP11_CONDITION_VARIABLE",
		"EA_HAVE_CPP11_MUTEX", "EA_HAVE_CPP11_THREAD", "EA_HAVE_CPP11_FUTURE", "EA_HAVE_CPP11_TYPE_TRAITS",
		"EA_HAVE_CPP11_TUPLES", "EA_HAVE_CPP11_REGEX", "EA_HAVE_CPP11_RANDOM", "EA_HAVE_CPP11_CHRONO",
		"EA_HAVE_CPP11_SCOPED_ALLOCATOR", "EA_HAVE_CPP11_INITIALIZER_LIST", "EA_HAVE_CPP11_SYSTEM_ERROR",
		"EA_HAVE_CPP11_TYPEINDEX", "EASTL_STD_ITERATOR_CATEGORY_ENABLED", "EASTL_STD_TYPE_TRAITS_AVAILABLE",
		"EASTL_MOVE_SEMANTICS_ENABLED", "EASTL_VARIADIC_TEMPLATES_ENABLED", "EASTL_VARIABLE_TEMPLATES_ENABLED",
		"EASTL_INLINE_VARIABLE_ENABLED", "EASTL_HAVE_CPP11_TYPE_TRAITS", "EASTL_INLINE_NAMESPACES_ENABLED",
		"EASTL_ALLOCATOR_EXPLICIT_ENABLED", "EA_DLL", "EASTL_USER_DEFINED_ALLOCATOR",
	--[[lc-core]]
		"UNICODE=1", "NOMINMAX=1", "_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1", "_CRT_SECURE_NO_WARNINGS=1",
		"_ENABLE_EXTENDED_ALIGNED_STORAGE=1", 
		
	{
		public = is_public
	})
end
