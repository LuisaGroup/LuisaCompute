set(LUISA_COMPUTE_OSL_SOURCES
        hint.cpp
        instruction.cpp
        literal.cpp
        shader.cpp
        symbol.cpp
        type.cpp
        oso_parser.cpp)

add_library(luisa-compute-osl SHARED ${LUISA_COMPUTE_OSL_SOURCES})
target_link_libraries(luisa-compute-osl PUBLIC luisa-compute-ast)

target_compile_definitions(luisa-compute-osl PRIVATE LC_OSL_EXPORT_DLL=1)

set_target_properties(luisa-compute-osl PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-osl)

luisa_compute_install(osl)
