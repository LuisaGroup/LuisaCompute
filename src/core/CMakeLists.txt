set(LUISA_COMPUTE_CORE_SOURCES
    logging.cpp
    logging.h
    memory.cpp memory.h
    concepts.h
    hash.cpp hash.h
    macro.h
    union.h
    platform.cpp platform.h
    mathematics.h
    atomic.h
    dynamic_module.cpp dynamic_module.h
    basic_types.cpp basic_types.h
    intrin.h)

find_package(Threads REQUIRED)

add_library(luisa-compute-core SHARED ${LUISA_COMPUTE_CORE_SOURCES})
target_link_libraries(luisa-compute-core PUBLIC
                      luisa-compute-include
                      luisa-compute-ext
                      Threads::Threads
                      ${CMAKE_DL_LIBS})
set_target_properties(luisa-compute-core PROPERTIES
                      WINDOWS_EXPORT_ALL_SYMBOLS ON
                      UNITY_BUILD ON)
