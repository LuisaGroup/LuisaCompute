# LLVMConfig.cmake may export the CRT mode used to build LLVM. Preserve the
# project's value so including LLVM never makes a later ABI check tautological.
set(_luisa_simd_had_msvc_runtime_library FALSE)
if (DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(_luisa_simd_had_msvc_runtime_library TRUE)
    set(_luisa_simd_project_msvc_runtime_library
            "${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif ()
find_package(LLVM CONFIG REQUIRED)
if (_luisa_simd_had_msvc_runtime_library)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
            "${_luisa_simd_project_msvc_runtime_library}")
else ()
    unset(CMAKE_MSVC_RUNTIME_LIBRARY)
endif ()
find_package(embree CONFIG REQUIRED)
if (EMBREE_TASKING_SYSTEM STREQUAL "TBB")
    # The backend explicitly quiesces Embree's oneTBB scheduler before the
    # dynamically loaded module is unloaded. A transitive libtbb dependency is
    # not enough here: the scheduler-handle API is used by simd_device.cpp.
    find_package(TBB CONFIG REQUIRED)
endif ()

if (LLVM_LINK_LLVM_DYLIB AND TARGET LLVM)
    set(_luisa_simd_llvm_libraries LLVM)
else ()
    llvm_map_components_to_libnames(_luisa_simd_llvm_libraries
            core
            support
            analysis
            passes
            executionengine
            orcjit
            nativecodegen)
endif ()

message(STATUS "Build SIMD CPU codegen core with LLVM ${LLVM_PACKAGE_VERSION}")

add_library(luisa-compute-simd-schedule STATIC
        schedule/block_barrier.cpp
        schedule/buffer_read_latency_hiding.cpp
        schedule/predicated_if_conversion.cpp
        schedule/loop_unswitch.cpp
        schedule/schedule_ir.cpp
        schedule/warp_uniformity.cpp
        schedule/xir_to_schedule.cpp)
target_compile_features(luisa-compute-simd-schedule PUBLIC cxx_std_20)
target_include_directories(luisa-compute-simd-schedule PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/schedule)
target_link_libraries(luisa-compute-simd-schedule PUBLIC
        luisa-compute-xir)
set_target_properties(luisa-compute-simd-schedule PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF)

set(LUISA_SIMD_LLVM_CODEGEN_SOURCES
        llvm/llvm_schedule_codegen.cpp
        llvm/llvm_schedule_codegen_cooperative.cpp
        llvm/llvm_schedule_emitter_bindless.cpp
        llvm/llvm_schedule_emitter.cpp
        llvm/llvm_schedule_emitter_arithmetic.cpp
        llvm/llvm_schedule_emitter_accel.cpp
        llvm/llvm_schedule_emitter_collectives.cpp
        llvm/llvm_schedule_emitter_cooperative.cpp
        llvm/llvm_schedule_emitter_control.cpp
        llvm/llvm_schedule_emitter_debug.cpp
        llvm/llvm_schedule_emitter_direct.cpp
        llvm/llvm_schedule_emitter_memory.cpp
        llvm/llvm_schedule_emitter_texture.cpp
        llvm/llvm_schedule_emitter_terminators.cpp
        llvm/llvm_schedule_emitter_local_regions.cpp
        llvm/llvm_schedule_emitter_predication.cpp
        llvm/llvm_schedule_emitter_state.cpp
        llvm/llvm_schedule_emitter_structured_loop.cpp
        llvm/llvm_schedule_emitter_ray_query_analysis.cpp
        llvm/llvm_schedule_emitter_ray_query.cpp
        llvm/llvm_schedule_emitter_ray_query_filter.cpp
        llvm/llvm_schedule_emitter_ray_query_pipeline.cpp
        llvm/llvm_schedule_emitter_ray_query_packet.cpp)

# Keep the hand-written code generator reviewable. Generated sources are not
# part of this list; a focused lowering unit that grows past this budget must
# be split by responsibility before more functionality is added to it.
foreach (_luisa_simd_codegen_source IN LISTS LUISA_SIMD_LLVM_CODEGEN_SOURCES)
    file(STRINGS
            "${CMAKE_CURRENT_SOURCE_DIR}/${_luisa_simd_codegen_source}"
            _luisa_simd_codegen_lines)
    list(LENGTH _luisa_simd_codegen_lines _luisa_simd_codegen_line_count)
    if (_luisa_simd_codegen_line_count GREATER 2000)
        message(FATAL_ERROR
                "SIMD LLVM source ${_luisa_simd_codegen_source} has "
                "${_luisa_simd_codegen_line_count} non-empty lines; split it "
                "before exceeding the 2000-line review budget.")
    endif ()
endforeach ()
unset(_luisa_simd_codegen_source)
unset(_luisa_simd_codegen_lines)
unset(_luisa_simd_codegen_line_count)

add_library(luisa-compute-simd-llvm STATIC
        ../common/llvm_native_math.cpp
        ../common/llvm_native_math_atan2.cpp
        ../common/llvm_native_math_fast_exp_log.cpp
        ../common/llvm_native_math_fast_inverse_trig.cpp
        ../common/llvm_native_math_fast_trig.cpp
        ../common/llvm_native_math_hyperbolic.cpp
        ../common/llvm_native_math_pow.cpp
        ../common/llvm_native_math_precise_exp_log.cpp
        ../common/llvm_native_math_precise_inverse_trig.cpp
        ../common/llvm_native_math_precise_trig.cpp
        ../common/llvm_native_math_range_reduction.cpp
        llvm/llvm_jit.cpp
        ${LUISA_SIMD_LLVM_CODEGEN_SOURCES}
        llvm/llvm_value_layout.cpp
        llvm/llvm_warp_collectives.cpp)
target_compile_features(luisa-compute-simd-llvm PUBLIC cxx_std_20)
target_include_directories(luisa-compute-simd-llvm
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/llvm
        ${CMAKE_CURRENT_SOURCE_DIR}/schedule)
target_include_directories(luisa-compute-simd-llvm SYSTEM PUBLIC
        ${LLVM_INCLUDE_DIRS})
target_link_libraries(luisa-compute-simd-llvm
        PUBLIC luisa-compute-simd-schedule
        PRIVATE ${_luisa_simd_llvm_libraries}
        embree)
target_compile_definitions(luisa-compute-simd-llvm
        PRIVATE
        ${LLVM_DEFINITIONS}
        LUISA_COMPUTE_SIMD_EMBREE_VERSION=${embree_VERSION_MAJOR})
set_target_properties(luisa-compute-simd-llvm PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF)

add_library(luisa-compute-simd-compiler STATIC
        simd_compiler.cpp)
target_compile_features(luisa-compute-simd-compiler PUBLIC cxx_std_20)
target_include_directories(luisa-compute-simd-compiler PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(luisa-compute-simd-compiler PUBLIC
        luisa-compute-simd-llvm)
set_target_properties(luisa-compute-simd-compiler PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        UNITY_BUILD OFF)

luisa_compute_add_backend(simd SOURCES
        ../fallback/fallback_texture.cpp
        runtime/simd_accel.cpp
        runtime/simd_accel_motion.cpp
        runtime/simd_bindless_array.cpp
        runtime/simd_buffer.cpp
        runtime/simd_curve.cpp
        runtime/simd_device.cpp
        runtime/simd_event.cpp
        runtime/simd_mesh.cpp
        runtime/simd_motion_instance.cpp
        runtime/simd_procedural_primitive.cpp
        runtime/simd_shader.cpp
        runtime/simd_stream.cpp
        runtime/simd_thread_pool.cpp
        runtime/simd_texture.cpp
        # Keep these append-only specializations last so existing narrow
        # runtime object order and hot-code layout remain stable.
        runtime/simd_accel_ray_query.cpp
        runtime/simd_accel_ray_query_triangle.cpp
        runtime/simd_accel_ray_query_status.cpp
        runtime/simd_accel_ray_query_filter_pipeline.cpp)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Keep the optional W8/W16 ray-query filter from contributing a small
    # .text.unlikely partition ahead of the established narrow hot text. On
    # GCC that 65-byte split shifts every narrow callback by 128 bytes and is
    # measurably harmful even though their machine code is otherwise equal.
    set_source_files_properties(
            runtime/simd_accel_ray_query.cpp
            runtime/simd_accel_ray_query_triangle.cpp
            runtime/simd_accel_ray_query_status.cpp
            runtime/simd_accel_ray_query_filter_pipeline.cpp
            PROPERTIES COMPILE_OPTIONS "-fno-reorder-blocks-and-partition")
endif ()
target_link_libraries(luisa-compute-backend-simd PRIVATE
        luisa-compute-simd-compiler
        embree)
if (EMBREE_TASKING_SYSTEM STREQUAL "TBB" AND TARGET TBB::tbb)
    target_link_libraries(luisa-compute-backend-simd PRIVATE TBB::tbb)
    target_compile_definitions(luisa-compute-backend-simd PRIVATE
            LUISA_COMPUTE_SIMD_HAS_TBB_SCHEDULER_HANDLE=1)
else ()
    target_compile_definitions(luisa-compute-backend-simd PRIVATE
            LUISA_COMPUTE_SIMD_HAS_TBB_SCHEDULER_HANDLE=0)
endif ()
target_compile_definitions(luisa-compute-backend-simd PRIVATE
        LUISA_COMPUTE_SIMD_EMBREE_VERSION=${embree_VERSION_MAJOR})
set_target_properties(luisa-compute-backend-simd PROPERTIES
        UNITY_BUILD OFF)
