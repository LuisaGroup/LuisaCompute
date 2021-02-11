set(LUISA_COMPUTE_CORE_SOURCES
    logging.h
    arena.h
    concepts.h
    type_info.cpp type_info.h
    hash.cpp hash.h
    macro_map.h
    union.h platform.h)

add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
target_link_libraries(luisa-compute-core PUBLIC luisa-compute-include luisa-compute-ext)
set_target_properties(luisa-compute-core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
