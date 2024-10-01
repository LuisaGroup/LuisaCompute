set(LUISA_COMPUTE_XIR_SOURCES
        argument.cpp
        basic_block.cpp
        constant.cpp
        function.cpp
        instruction.cpp
        metadata.cpp
        pool.cpp
        shared.cpp
        use.cpp
        user.cpp
        value.cpp

        # derived instructions
        instructions/branch.cpp
        instructions/loop.cpp

        # derived metadata
        metadata/location.cpp
)

add_library(luisa-compute-xir SHARED ${LUISA_COMPUTE_XIR_SOURCES})
target_link_libraries(luisa-compute-xir PUBLIC luisa-compute-ast)

target_compile_definitions(luisa-compute-xir PRIVATE LC_XIR_EXPORT_DLL=1)

set_target_properties(luisa-compute-xir PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-xir)

luisa_compute_install(xir)
