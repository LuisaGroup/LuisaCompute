set(LUISA_COMPUTE_OSL_SOURCES
        hint.cpp
        instruction.cpp
        literal.cpp
        shader.cpp
        symbol.cpp
        type.cpp
        oso_parser.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(_LUISA_OSL_LIBRARY_TYPE STATIC)
else ()
    set(_LUISA_OSL_LIBRARY_TYPE SHARED)
endif ()
add_library(luisa-compute-osl
        ${_LUISA_OSL_LIBRARY_TYPE} ${LUISA_COMPUTE_OSL_SOURCES})
target_link_libraries(luisa-compute-osl PUBLIC luisa-compute-ast)

target_compile_definitions(luisa-compute-osl PRIVATE LUISA_OSL_EXPORT_DLL=1)

set_target_properties(luisa-compute-osl PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-osl)

luisa_compute_install(osl)
