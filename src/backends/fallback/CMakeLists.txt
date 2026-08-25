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
            ../common/llvm_native_math_hyperbolic.cpp
            ../common/llvm_native_math_pow.cpp
            ../common/llvm_native_math_precise_exp_log.cpp
            ../common/llvm_native_math_precise_inverse_trig.cpp
            ../common/llvm_native_math_precise_trig.cpp
            ../common/llvm_native_math_range_reduction.cpp
            ../common/default_binary_io.cpp
    )

    luisa_compute_add_backend(fallback SOURCES ${LUISA_FALLBACK_BACKEND_SOURCES})

    # The fallback JIT consumes this module through LLVM's C++ API. LLVM
    # bitcode is not stable across downstream distributions, even when their
    # major versions match, so generate it with the clang/llvm-as pair from the
    # exact LLVM package linked below instead of embedding a checked-in blob.
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    find_program(_luisa_fallback_clang
            NAMES clang++
            HINTS "${LLVM_TOOLS_BINARY_DIR}"
            NO_DEFAULT_PATH REQUIRED NO_CACHE)
    find_program(_luisa_fallback_llvm_as
            NAMES llvm-as
            HINTS "${LLVM_TOOLS_BINARY_DIR}"
            NO_DEFAULT_PATH REQUIRED NO_CACHE)

    if (CMAKE_CXX_COMPILER_TARGET)
        set(_luisa_fallback_target_triple
                "${CMAKE_CXX_COMPILER_TARGET}")
    else ()
        string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}"
                _luisa_fallback_system_processor)
        if (_luisa_fallback_system_processor MATCHES
                "^(x86_64|amd64|x64)$")
            set(_luisa_fallback_target_arch x86_64)
        elseif (_luisa_fallback_system_processor MATCHES
                "^(aarch64|arm64)$")
            set(_luisa_fallback_target_arch aarch64)
        else ()
            message(FATAL_ERROR
                    "Unsupported fallback target architecture: "
                    "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        if (WIN32)
            set(_luisa_fallback_target_triple
                    "${_luisa_fallback_target_arch}-pc-windows-msvc")
        elseif (APPLE)
            set(_luisa_fallback_target_triple
                    "${_luisa_fallback_target_arch}-apple-darwin")
        elseif (UNIX)
            set(_luisa_fallback_target_triple
                    "${_luisa_fallback_target_arch}-unknown-linux-gnu")
        else ()
            message(FATAL_ERROR
                    "Unsupported fallback target system: "
                    "${CMAKE_SYSTEM_NAME}")
        endif ()
    endif ()
    if (APPLE)
        # CMake's deployment target is part of the host ABI contract. Without
        # it, clang chooses an implementation-defined minimum OS in the
        # embedded module that can disagree with the enclosing binary.
        if (CMAKE_OSX_DEPLOYMENT_TARGET)
            set(_luisa_fallback_deployment_target
                    "${CMAKE_OSX_DEPLOYMENT_TARGET}")
        else ()
            set(_luisa_fallback_deployment_target 11.0)
        endif ()
    else ()
        set(_luisa_fallback_deployment_target "")
    endif ()

    set(_luisa_fallback_builtin_source
            "${CMAKE_CURRENT_LIST_DIR}/fallback_builtin/fallback_device_api_wrappers.cpp")
    set(_luisa_fallback_builtin_generator
            "${CMAKE_CURRENT_LIST_DIR}/fallback_builtin/generate_and_embed_fallback_device_lib.py")
    set(_luisa_fallback_builtin_dir
            "${CMAKE_CURRENT_BINARY_DIR}/fallback_builtin")
    set(_luisa_fallback_builtin_ll
            "${_luisa_fallback_builtin_dir}/fallback_device_api_wrappers.ll")
    set(_luisa_fallback_builtin_bc
            "${_luisa_fallback_builtin_dir}/fallback_device_api_wrappers.bc")
    set(_luisa_fallback_symbol_map
            "${_luisa_fallback_builtin_dir}/fallback_device_api_map_symbols.generated.inl.h")
    set(_luisa_fallback_builtin_embedded_cpp
            "${_luisa_fallback_builtin_dir}/fallback_device_api_wrappers_embedded.cpp")
    set(_luisa_fallback_builtin_embedded_h
            "${_luisa_fallback_builtin_dir}/fallback_device_api_wrappers_embedded.h")

    add_custom_command(
            OUTPUT
            "${_luisa_fallback_builtin_ll}"
            "${_luisa_fallback_builtin_bc}"
            "${_luisa_fallback_symbol_map}"
            COMMAND "${Python3_EXECUTABLE}"
            "${_luisa_fallback_builtin_generator}"
            --clang "${_luisa_fallback_clang}"
            --llvm-as "${_luisa_fallback_llvm_as}"
            --source "${_luisa_fallback_builtin_source}"
            --target "${_luisa_fallback_target_triple}"
            --deployment-target "${_luisa_fallback_deployment_target}"
            --output-ll "${_luisa_fallback_builtin_ll}"
            --output-bc "${_luisa_fallback_builtin_bc}"
            --output-symbol-map "${_luisa_fallback_symbol_map}"
            DEPENDS
            "${_luisa_fallback_builtin_generator}"
            "${_luisa_fallback_builtin_source}"
            "${CMAKE_CURRENT_LIST_DIR}/fallback_device_api.h"
            COMMENT
            "[Fallback] Compiling device API with ${_luisa_fallback_clang}"
            VERBATIM)
    add_custom_command(
            OUTPUT
            "${_luisa_fallback_builtin_embedded_cpp}"
            "${_luisa_fallback_builtin_embedded_h}"
            COMMAND $<TARGET_FILE:luisa_embed_device_lib>
            "${_luisa_fallback_builtin_bc}"
            --unsigned --prefix luisa_compute_
            -o "${_luisa_fallback_builtin_embedded_cpp}"
            -h "${_luisa_fallback_builtin_embedded_h}"
            DEPENDS
            luisa_embed_device_lib
            "${_luisa_fallback_builtin_bc}"
            COMMENT "[Fallback] Embedding device API bitcode"
            VERBATIM)
    add_custom_target(luisa-compute-backend-fallback-embed-builtins
            DEPENDS
            "${_luisa_fallback_builtin_ll}"
            "${_luisa_fallback_builtin_bc}"
            "${_luisa_fallback_symbol_map}"
            "${_luisa_fallback_builtin_embedded_cpp}"
            "${_luisa_fallback_builtin_embedded_h}")
    add_dependencies(luisa-compute-backend-fallback
            luisa-compute-backend-fallback-embed-builtins)
    set_source_files_properties(
            "${_luisa_fallback_builtin_embedded_cpp}"
            "${_luisa_fallback_builtin_embedded_h}"
            "${_luisa_fallback_symbol_map}"
            PROPERTIES GENERATED TRUE SKIP_UNITY_BUILD_INCLUSION TRUE)
    target_sources(luisa-compute-backend-fallback PRIVATE
            "${_luisa_fallback_builtin_embedded_cpp}"
            "${_luisa_fallback_builtin_embedded_h}"
            "${_luisa_fallback_symbol_map}")
    target_include_directories(luisa-compute-backend-fallback BEFORE PRIVATE
            "${_luisa_fallback_builtin_dir}")

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
