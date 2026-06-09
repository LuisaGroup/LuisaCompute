set(LUISA_COMPUTE_CORO_SOURCES
        coro_compile.cpp
        coro_graph.cpp
        coro_stub.cpp
        schedulers/state_machine.cpp
        schedulers/wavefront.cpp
        schedulers/persistent.cpp)

add_library(luisa-compute-coro SHARED ${LUISA_COMPUTE_CORO_SOURCES})
target_link_libraries(luisa-compute-coro PUBLIC luisa-compute-dsl luisa-compute-xir)
target_compile_definitions(luisa-compute-coro
        PRIVATE LUISA_CORO_EXPORT_DLL=1)
set_target_properties(luisa-compute-coro PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-coro)
luisa_compute_install(coro SOURCES ${LUISA_COMPUTE_CORO_SOURCES})
