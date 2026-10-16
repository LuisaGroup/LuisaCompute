# luisa-fallback-rtx: the software (hand-built two-level LBVH) ray-tracing
# fallback shared by the backends that have no hardware ray tracing, or that the
# user asked to run in fallback mode.  Backend-independent: it only needs the
# runtime and the DSL, and it talks to the backend through the `DeviceInterface`
# it is handed (fallback_rtx.h).
#
# This subdirectory is added by src/backends/common/CMakeLists.txt when at least
# one of the cuda/dx/vk backends is enabled.

add_library(luisa-fallback-rtx STATIC
        fallback_rtx_sort.cpp
        fallback_rtx_storage.cpp
        fallback_rtx_blas.cpp
        fallback_rtx_tlas.cpp
        fallback_rtx_device.cpp
        fallback_rtx_layout_contract.cpp)
set_target_properties(luisa-fallback-rtx PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(luisa-fallback-rtx PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(luisa-fallback-rtx PUBLIC
        luisa-compute-runtime
        luisa-compute-dsl)

# The self-check: creates a real device, builds a two-level LBVH through the
# library's public API only, and validates it.  It links the whole frontend so a
# device can be created at runtime, exactly like an example does.
add_executable(fallback_rtx_selfcheck fallback_rtx_selfcheck.cpp)
target_link_libraries(fallback_rtx_selfcheck PRIVATE
        luisa-fallback-rtx
        luisa-compute-runtime
        luisa-compute-dsl
        luisa-compute-vstl
        luisa-compute-backends)
