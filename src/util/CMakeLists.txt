set(LUISA_COMPUTE_UTIL_SOURCES
    arena.cpp arena.h
    dynamic_module.cpp dynamic_module.h
    spin_mutex.h)

if (WIN32)
    list(APPEND LUISA_COMPUTE_UTIL_SOURCES
	vstl.cpp 
	Log.cpp
	ObjectPtr.cpp
    StackAllocator.cpp)
endif ()

add_library(luisa-compute-util SHARED ${LUISA_COMPUTE_UTIL_SOURCES})
target_compile_definitions(luisa-compute-util PRIVATE LUISA_COMPUTE_UTIL_INTERNAL PUBLIC LUISA_COMPUTE_UTIL_EXTERNAL)
target_link_libraries(luisa-compute-util PUBLIC luisa-compute-core PRIVATE mimalloc-static)
set_target_properties(luisa-compute-util PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
