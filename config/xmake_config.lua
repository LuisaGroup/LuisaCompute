function add_lc_includes(rootDir, isPublic)
	-- spdlog
	add_defines("SPDLOG_NO_EXCEPTIONS", "SPDLOG_NO_THREAD_ID", "SPDLOG_DISABLE_DEFAULT_LOGGER", "SPDLOG_COMPILED_LIB",
					"FMT_SHARED", "SPDLOG_SHARED_LIB", "FMT_CONSTEVAL=constexpr", "FMT_USE_CONSTEXPR=1", "FMT_EXCEPTIONS=0", {
						public = isPublic
					})
	add_includedirs(rootDir .. "/src/ext/spdlog/include", {
		public = isPublic
	})
	-- mimalloc
	add_defines("MI_SHARED_LIB", {
		public = isPublic
	})
	add_includedirs(rootDir .. "/src/ext/EASTL/packages/mimalloc/include", {
		public = isPublic
	})
	-- eastl
	add_defines("EA_PRAGMA_ONCE_SUPPORTED", "EASTL_ASSERT_ENABLED=0", "EA_HAVE_CPP11_CONTAINERS", "EA_HAVE_CPP11_ATOMIC",
					"EA_HAVE_CPP11_CONDITION_VARIABLE", "EA_HAVE_CPP11_MUTEX", "EA_HAVE_CPP11_THREAD", "EA_HAVE_CPP11_FUTURE",
					"EA_HAVE_CPP11_TYPE_TRAITS", "EA_HAVE_CPP11_TUPLES", "EA_HAVE_CPP11_REGEX", "EA_HAVE_CPP11_RANDOM",
					"EA_HAVE_CPP11_CHRONO", "EA_HAVE_CPP11_SCOPED_ALLOCATOR", "EA_HAVE_CPP11_INITIALIZER_LIST",
					"EA_HAVE_CPP11_SYSTEM_ERROR", "EA_HAVE_CPP11_TYPEINDEX", "EASTL_STD_ITERATOR_CATEGORY_ENABLED",
					"EASTL_STD_TYPE_TRAITS_AVAILABLE", "EASTL_MOVE_SEMANTICS_ENABLED", "EASTL_VARIADIC_TEMPLATES_ENABLED",
					"EASTL_VARIABLE_TEMPLATES_ENABLED", "EASTL_INLINE_VARIABLE_ENABLED", "EASTL_HAVE_CPP11_TYPE_TRAITS",
					"EASTL_INLINE_NAMESPACES_ENABLED", "EASTL_ALLOCATOR_EXPLICIT_ENABLED", "EA_DLL", "EASTL_USER_DEFINED_ALLOCATOR", {
						public = isPublic
					})
	add_includedirs(rootDir .. "/src/ext/EASTL/include", rootDir .. "/src/ext/EASTL/packages/EABase/include/Common", {
		public = isPublic
	})
	-- lc-core
	add_defines("UNICODE=1", "NOMINMAX=1", "_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS=1", "_CRT_SECURE_NO_WARNINGS=1",
					"_ENABLE_EXTENDED_ALIGNED_STORAGE=1", {
						public = isPublic
					})
	add_includedirs(rootDir .. "/src", rootDir .. "/src/ext/xxHash", rootDir .. "/src/ext/parallel-hashmap", {
		public = isPublic
	})
end
