set(LUISA_COMPUTE_CORO_SOURCES
        coro_frame.cpp
        coro_frame_buffer.cpp
        coro_frame_desc.cpp
        coro_frame_smem.cpp
        coro_func.cpp
        coro_graph.cpp
        schedulers/persistent_threads.cpp
        schedulers/state_machine.cpp
        schedulers/wavefront.cpp)

add_library(luisa-compute-coro SHARED ${LUISA_COMPUTE_CORO_SOURCES})
target_link_libraries(luisa-compute-coro PUBLIC
        luisa-compute-ast
        luisa-compute-dsl
        luisa-compute-runtime
        luisa-compute-xir)
target_precompile_headers(luisa-compute-coro PRIVATE pch.h)

target_compile_definitions(luisa-compute-coro PRIVATE LUISA_CORO_EXPORT_DLL=1)
set_target_properties(luisa-compute-coro PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-coro)

luisa_compute_install(coro SOURCES ${LUISA_COMPUTE_CORO_SOURCES})
