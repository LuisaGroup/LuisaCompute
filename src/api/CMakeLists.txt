set(LUISA_COMPUTE_API_SOURCES
        runtime.cpp runtime.h
        type.cpp type.h logging.cpp logging.h)
add_library(luisa-compute-api SHARED ${LUISA_COMPUTE_API_SOURCES})
target_link_libraries(luisa-compute-api PUBLIC
        luisa-compute-ast
        luisa-compute-runtime
        luisa-compute-serialize
        luisa-compute-compile)
set_target_properties(luisa-compute-api PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        UNITY_BUILD ON)
