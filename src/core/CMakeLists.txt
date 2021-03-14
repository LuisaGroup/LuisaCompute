set(LUISA_COMPUTE_CORE_SOURCES
    logging.cpp
    logging.h
    memory.h
    concepts.h
    hash.cpp hash.h
    macro.h
    union.h
    platform.h
    mathematics.h
    atomic.h
    dynamic_dll.h
    dynamic_dll.cpp
    basic_types.h)

add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
target_link_libraries(luisa-compute-core PUBLIC luisa-compute-include luisa-compute-ext)
set_target_properties(luisa-compute-core PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
