set(LUISA_COMPUTE_CORE_SOURCES
    context.cpp context.h
    logging.cpp logging.h
    arena.cpp arena.h
    concepts.cpp concepts.h
    type_info.cpp type_info.h
    hash.cpp hash.h
    macro_map.h)

add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
target_link_libraries(luisa-compute-core PUBLIC luisa-compute-include luisa-compute-ext)
