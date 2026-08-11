# LLVMConfig.cmake exports its build-time CMAKE_MSVC_RUNTIME_LIBRARY into the
# including directory. Preserve the project's value across this dependency
# probe: otherwise a later LLVM consumer could mistake LLVM's leaked setting
# for the project's own CRT selection and turn its ABI check into a tautology.
set(_luisa_fallback_had_msvc_runtime_library FALSE)
if (DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(_luisa_fallback_had_msvc_runtime_library TRUE)
    set(_luisa_fallback_project_msvc_runtime_library
            "${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif ()
find_package(LLVM CONFIG)
if (_luisa_fallback_had_msvc_runtime_library)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
            "${_luisa_fallback_project_msvc_runtime_library}")
else ()
    unset(CMAKE_MSVC_RUNTIME_LIBRARY)
endif ()
find_package(embree CONFIG)

if (NOT embree_FOUND)
    message(WARNING "Embree not found. Fallback backend will not be built.")
endif ()

if (NOT LLVM_FOUND)
    message(WARNING "LLVM not found. Fallback backend will not be built.")
endif ()

if (LLVM_FOUND AND embree_FOUND)

    message(STATUS "Build with fallback backend (LLVM ${LLVM_VERSION}, Embree ${embree_VERSION})")

    set(LUISA_FALLBACK_BACKEND_SOURCES
            fallback_device.cpp
            fallback_device_api.cpp
            fallback_device_api_ir_module.cpp
            fallback_event.cpp
            fallback_command_queue.cpp
            fallback_bindless_array.cpp
            fallback_stream.cpp
            fallback_texture.cpp
            fallback_prim.cpp
            fallback_mesh.cpp
            fallback_curve.cpp
            fallback_proc_prim.cpp
            fallback_motion_instance.cpp
            fallback_accel.cpp
            fallback_texture_bc.cpp
            fallback_codegen.cpp
            fallback_shader.cpp
            fallback_buffer.cpp
            fallback_swapchain.cpp
            fallback_tex_compress.cpp
            ../common/llvm_native_math.cpp
            ../common/llvm_native_math_atan2.cpp
            ../common/llvm_native_math_fast_exp_log.cpp
            ../common/llvm_native_math_fast_inverse_trig.cpp
            ../common/llvm_native_math_fast_trig.cpp
            ../common/llvm_native_math_precise_exp_log.cpp
            ../common/llvm_native_math_precise_inverse_trig.cpp
            ../common/llvm_native_math_precise_trig.cpp
            ../common/llvm_native_math_range_reduction.cpp
            ../common/default_binary_io.cpp
    )

    luisa_compute_add_backend(fallback SOURCES ${LUISA_FALLBACK_BACKEND_SOURCES})

    # exclude codegen from unity build since it's too big
    set_source_files_properties(fallback_codegen.cpp PROPERTIES UNITY_BUILD OFF)
    target_link_libraries(luisa-compute-backend-fallback PRIVATE
            luisa-compute-xir
            luisa-compute-vulkan-swapchain
            embree)

    target_compile_definitions(luisa-compute-backend-fallback PRIVATE
            LUISA_COMPUTE_FALLBACK_EMBREE_VERSION=${embree_VERSION_MAJOR})

    option(LUISA_COMPUTE_USE_SYSTEM_PARALLEL_FOR "Enable system parallel_for" OFF)
    if (LUISA_COMPUTE_USE_SYSTEM_PARALLEL_FOR)
        target_compile_definitions(luisa-compute-backend-fallback PRIVATE LUISA_COMPUTE_USE_SYSTEM_PARALLEL_FOR=1)
        # use libdispatch or TBB for parallel_for on non-Apple Unix systems if available
        if (UNIX AND NOT APPLE)
            # libdispatch
            find_library(DISPATCH_LIB dispatch)
            find_path(DISPATCH_INCLUDE_DIR dispatch/dispatch.h)
            if (DISPATCH_LIB AND DISPATCH_INCLUDE_DIR)
                target_compile_definitions(luisa-compute-backend-fallback PRIVATE LUISA_COMPUTE_ENABLE_LIBDISPATCH=1)
                target_link_libraries(luisa-compute-backend-fallback PRIVATE ${DISPATCH_LIB})
                target_include_directories(luisa-compute-backend-fallback PRIVATE ${DISPATCH_INCLUDE_DIR})
            else ()
                 find_package(TBB CONFIG)
                if (TBB_FOUND)
                    target_compile_definitions(luisa-compute-backend-fallback PRIVATE LUISA_COMPUTE_ENABLE_TBB=1)
                    target_link_libraries(luisa-compute-backend-fallback PRIVATE TBB::tbb)
                else ()
                    message(WARNING "TBB and libdispatch not found. Performance of the fallback backend may be affected.")
                endif ()
            endif ()
        endif ()
    endif ()

    # LLVM
    luisa_compute_link_llvm_into_backend(fallback REQUIRED
            COMPONENTS core executionengine support orcjit nativecodegen irreader passes analysis coroutines)

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "LLVM or Embree not found for the fallback backend.")
else ()
    message(WARNING "LLVM or Embree not found. Fallback backend will not be built.")
endif ()
