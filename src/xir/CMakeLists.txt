set(LUISA_COMPUTE_XIR_SOURCES
        basic_block.cpp
        constant.cpp
        function.cpp
        instruction.cpp
        metadata.cpp
        name.cpp
        pool.cpp
        use.cpp
        user.cpp
        value.cpp
        variable.cpp

        # derived instructions
        instructions/branch.cpp
        instructions/break.cpp
        instructions/call.cpp
        instructions/cast.cpp
        instructions/comment.cpp
        instructions/continue.cpp
        instructions/gep.cpp
        instructions/intrinsic.cpp
        instructions/load.cpp
        instructions/loop.cpp
        instructions/phi.cpp
        instructions/print.cpp
        instructions/return.cpp
        instructions/store.cpp
        instructions/switch.cpp

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
