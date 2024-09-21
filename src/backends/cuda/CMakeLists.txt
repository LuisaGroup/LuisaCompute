find_package(CUDAToolkit 11.7 COMPONENTS cuda_driver nvrtc_static)

if (CUDAToolkit_FOUND)
    message(STATUS "Build with CUDA backend: ${CUDAToolkit_VERSION}")

    set(LUISA_COMPUTE_CUDA_SOURCES
            cuda_error.h
            cuda_bindless_array.cpp cuda_bindless_array.h
            cuda_buffer.cpp cuda_buffer.h
            cuda_stream.cpp cuda_stream.h
            cuda_device.cpp cuda_device.h
            cuda_event.cpp cuda_event.h
            cuda_host_buffer_pool.cpp cuda_host_buffer_pool.h
            cuda_command_encoder.cpp cuda_command_encoder.h
            cuda_texture.cpp cuda_texture.h
            cuda_codegen_ast.cpp cuda_codegen_ast.h
            cuda_compiler.cpp cuda_compiler.h
            cuda_accel.cpp cuda_accel.h
            cuda_primitive.cpp cuda_primitive.h
            cuda_procedural_primitive.cpp cuda_procedural_primitive.h
            cuda_curve.cpp cuda_curve.h
            cuda_mesh.cpp cuda_mesh.h
            cuda_motion_instance.cpp cuda_motion_instance.h
            cuda_shader.cpp cuda_shader.h
            cuda_shader_metadata.cpp cuda_shader_metadata.h
            cuda_shader_native.cpp cuda_shader_native.h
            cuda_shader_optix.cpp cuda_shader_optix.h
            cuda_shader_printer.cpp cuda_shader_printer.h
            cuda_swapchain.cpp cuda_swapchain.h
            cuda_callback_context.h
            cuda_builtin_embedded.cpp cuda_builtin_embedded.h
            optix_api.cpp optix_api.h
            default_binary_io.cpp)

    luisa_compute_add_backend(cuda SOURCES ${LUISA_COMPUTE_CUDA_SOURCES})

    # extensions
    target_sources(luisa-compute-backend-cuda PRIVATE
            extensions/cuda_pinned_memory.cpp extensions/cuda_pinned_memory.h
            extensions/cuda_dstorage.cpp extensions/cuda_dstorage.h
            extensions/cuda_denoiser.cpp extensions/cuda_denoiser.h)

    target_link_libraries(luisa-compute-backend-cuda PRIVATE
            CUDA::cuda_driver
            luisa-compute-vulkan-swapchain
            reproc reproc++)
    target_precompile_headers(luisa-compute-backend-cuda PRIVATE pch.h)

    find_library(CUDA_DEVICE_RUNTIME_LIBRARY cudadevrt ${CUDAToolkit_LIBRARY_DIR})
    add_custom_command(TARGET luisa-compute-backend-cuda POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CUDA_DEVICE_RUNTIME_LIBRARY}"
            $<TARGET_FILE_DIR:luisa-compute-core>)
    install(FILES "${CUDA_DEVICE_RUNTIME_LIBRARY}" DESTINATION ${CMAKE_INSTALL_BINDIR})

    # nvrtc standalone compiler
    add_executable(luisa-cuda-nvrtc-standalone-compiler cuda_nvrtc_compiler.cpp)
    target_link_libraries(luisa-cuda-nvrtc-standalone-compiler PRIVATE CUDA::nvrtc_static)
    set_target_properties(luisa-cuda-nvrtc-standalone-compiler PROPERTIES OUTPUT_NAME "luisa_nvrtc")
    if (UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_link_options(luisa-cuda-nvrtc-standalone-compiler PRIVATE -stdlib=libstdc++)
        target_compile_options(luisa-cuda-nvrtc-standalone-compiler PRIVATE -stdlib=libstdc++)
    endif ()
    if (WIN32)
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_options(luisa-cuda-nvrtc-standalone-compiler PRIVATE
                    "LINKER:/FORCE:MULTIPLE"
                    "LINKER:/OPT:REF")
        endif ()
        target_link_libraries(luisa-cuda-nvrtc-standalone-compiler PRIVATE Ws2_32)
    endif ()
    add_dependencies(luisa-compute-backend-cuda luisa-cuda-nvrtc-standalone-compiler)
    install(TARGETS luisa-cuda-nvrtc-standalone-compiler RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    if (WIN32)
        target_link_libraries(luisa-compute-backend-cuda PRIVATE cfgmgr32)
    endif ()

    if (TARGET luisa-compute-oidn-ext)
        target_link_libraries(luisa-compute-backend-cuda PRIVATE luisa-compute-oidn-ext)
    endif ()

    # nvCOMP
    if (LUISA_COMPUTE_DOWNLOAD_NVCOMP)
        if (WIN32)
            set(NVCOMP_PLATFORM "windows")
            set(NVCOMP_EXT "zip")
        else ()
            set(NVCOMP_PLATFORM "x86_64")
            set(NVCOMP_EXT "tgz")
        endif ()
        set(NVCOMP_VERSION "3.0.4")
        set(NVCOMP_DOWNLOAD_URL "https://developer.download.nvidia.com/compute/nvcomp/${NVCOMP_VERSION}/local_installers/nvcomp_${NVCOMP_VERSION}_${NVCOMP_PLATFORM}_${CUDAToolkit_VERSION_MAJOR}.x.${NVCOMP_EXT}")
        message(STATUS "Downloading nvcomp from ${NVCOMP_DOWNLOAD_URL}")
        include(FetchContent)
        FetchContent_Declare(nvcomp URL ${NVCOMP_DOWNLOAD_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
        FetchContent_MakeAvailable(nvcomp)
        FetchContent_GetProperties(nvcomp)
        set(nvcomp_DIR ${nvcomp_SOURCE_DIR}/lib/cmake/nvcomp)
    endif ()

    find_package(nvcomp CONFIG)
    set(_NVCOMP_WARNING_MSG "nvCOMP not found. The CUDA backend will be built without GPU compression support. You may want to set the LUISA_COMPUTE_DOWNLOAD_NVCOMP option to ON to let LuisaCompute automatically download it.")
    if (NOT nvcomp_FOUND)
        message(WARNING ${_NVCOMP_WARNING_MSG})
    else ()
        if (${nvcomp_VERSION} VERSION_LESS "3.0.0")
            message(WARNING ${_NVCOMP_WARNING_MSG})
        else ()
            target_compile_definitions(luisa-compute-backend-cuda PRIVATE LUISA_COMPUTE_ENABLE_NVCOMP=1)
            set(NVCOMP_TARGETS
                    nvcomp::nvcomp
                    nvcomp::nvcomp_bitcomp
                    nvcomp::nvcomp_gdeflate)
            target_link_libraries(luisa-compute-backend-cuda PRIVATE ${NVCOMP_TARGETS})
            foreach (target ${NVCOMP_TARGETS})
                add_custom_command(TARGET luisa-compute-backend-cuda POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:${target}>
                        $<TARGET_FILE_DIR:luisa-compute-core>)
                install(FILES $<TARGET_FILE:${target}> DESTINATION ${CMAKE_INSTALL_BINDIR})
            endforeach ()
        endif ()
    endif ()

    # CUB LuisaCompute integration
    if (LUISA_COMPUTE_ENABLE_CUDA_EXT_LCUB)
        add_subdirectory(lcub)
    endif ()

    # NVTT
    if (NOT NVTT_DIR)
        set(_SEARCH_PATHS
                "${CMAKE_CURRENT_LIST_DIR}"
                "${CMAKE_CURRENT_LIST_DIR}/.."
                "${CMAKE_CURRENT_LIST_DIR}/../nvtt"
                "C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools"
                "$ENV{NVTT_DIR}")

        foreach (_PATH ${_SEARCH_PATHS})
            file(GLOB _DLL_POSSIBILITIES "${_PATH}/nvtt*.dll" "${_PATH}/libnvtt.so.*")
            if (_DLL_POSSIBILITIES) # If this folder contains a DLL matching the NVTT DLL pattern
                set(NVTT_DIR "${_PATH}")
                break ()
            endif ()
        endforeach ()

        if (NOT NVTT_DIR)
            message(WARNING "NVTT not found! Please install NVTT from https://developer.nvidia.com/nvidia-texture-tools-exporter and set the CMake NVTT_DIR variable to the folder containing nvtt*.dll (e.g. C:\\Program Files\\NVIDIA Corporation\\NVIDIA Texture Tools) .")
        endif ()
    endif ()

    # Get the NVTT shared library name.
    file(GLOB _NVTT_SL_POSSIBILITIES "${NVTT_DIR}/nvtt*.dll" "${NVTT_DIR}/libnvtt.so.*")
    if (NOT _NVTT_SL_POSSIBILITIES)
        message(WARNING "NVTT_DIR didn't contain an NVTT shared library of the form nvtt*.dll or libnvtt.so.*! Is NVTT_DIR set correctly? NVTT_DIR was ${NVTT_DIR}")
        return()
    else ()
        list(LENGTH _NVTT_SL_POSSIBILITIES _NVTT_SL_POSSIBILITIES_LEN)
        math(EXPR _NVTT_SL_IDX ${_NVTT_SL_POSSIBILITIES_LEN}-1)
        list(GET _NVTT_SL_POSSIBILITIES ${_NVTT_SL_IDX} _NVTT_SL)
    endif ()

    # Find the NVTT linker library on Windows.
    if (WIN32)
        if (NOT NVTT_LIB)
            file(GLOB _NVTT_LIB_ALL "${NVTT_DIR}/lib/x64-v*/nvtt*.lib")
            if (NOT _NVTT_LIB_ALL)
                message(WARNING "Found nvtt.dll in ${NVTT_DIR}, but was unable to find nvtt.lib in ${NVTT_DIR}/lib/... ! Please check the NVTT directory and this CMake script to see if the path is correct.")
            else ()
                list(LENGTH _NVTT_LIB_ALL _NVTT_LIB_LEN)
                math(EXPR _NVTT_LIB_IDX ${_NVTT_LIB_LEN}-1)
                list(GET _NVTT_LIB_ALL ${_NVTT_LIB_IDX} NVTT_LIB)
            endif ()
        endif ()
    endif ()

    if (_NVTT_SL)
        message(STATUS "Found NVTT shared library: ${_NVTT_SL}")
        # get include directories
        get_filename_component(_NVTT_SL_DIR "${_NVTT_SL}" DIRECTORY)
        if (EXISTS "${_NVTT_SL_DIR}/include/nvtt/nvtt.h" AND
                EXISTS "${_NVTT_SL_DIR}/include/nvtt/nvtt_wrapper.h" AND
                EXISTS "${_NVTT_SL_DIR}/include/nvtt/nvtt_lowlevel.h")
            set(NVTT_INCLUDE_DIR "${_NVTT_SL_DIR}/include")
        else ()
            message(WARNING "Unable to find NVTT include directory! NVTT shared library directory was ${_NVTT_SL_DIR}")
        endif ()
        # parse the NVTT shared library name to get the version number
        if (WIN32) # nvtt*.dll
            if (NVTT_LIB) # necessary on Windows
                string(REGEX MATCH "nvtt([0-9]+).dll" _NVTT_SL_VERSION "${_NVTT_SL}")
                if (NOT _NVTT_SL_VERSION OR _NVTT_SL_VERSION LESS 30200)
                    message(WARNING "Unable to parse the NVTT shared library name to get the version number! NVTT shared library name was ${_NVTT_SL}")
                else ()
                    set(NVTT_VERSION ${CMAKE_MATCH_1})
                    set(NVTT_LINKER_LIB ${NVTT_LIB})
                    set(NVTT_RUNTIME_LIB ${_NVTT_SL})
                endif ()
            endif ()
        else () # libnvtt.so.*
            string(REGEX MATCH "libnvtt.so.([0-9]+)" _NVTT_SL_VERSION "${_NVTT_SL}")
            if (NOT _NVTT_SL_VERSION OR _NVTT_SL_VERSION LESS 30200)
                message(WARNING "Unable to parse the NVTT shared library name to get the version number! NVTT shared library name was ${_NVTT_SL}")
            else ()
                set(NVTT_VERSION ${CMAKE_MATCH_1})
                set(NVTT_LINKER_LIB ${_NVTT_SL})
                set(NVTT_RUNTIME_LIB ${_NVTT_SL})
            endif ()
        endif ()
        if (NVTT_INCLUDE_DIR AND NVTT_VERSION AND NVTT_RUNTIME_LIB AND NVTT_LINKER_LIB)
            message(STATUS "Found NVTT ${NVTT_VERSION}\n NVTT_INCLUDE_DIR: ${NVTT_INCLUDE_DIR}\n NVTT_RUNTIME_LIB: ${NVTT_RUNTIME_LIB}\n NVTT_LINKER_LIB: ${NVTT_LINKER_LIB}")
            target_compile_definitions(luisa-compute-backend-cuda PRIVATE LUISA_COMPUTE_ENABLE_NVTT=1)
            target_include_directories(luisa-compute-backend-cuda PRIVATE ${NVTT_INCLUDE_DIR})
            target_link_libraries(luisa-compute-backend-cuda PRIVATE "${NVTT_LINKER_LIB}")
            target_sources(luisa-compute-backend-cuda PRIVATE
                    extensions/cuda_texture_compression.cpp
                    extensions/cuda_texture_compression.h)
            # The misplaced [[deprecated]] attributes in NVTT result in an error with Clang so we disable them.
            target_compile_definitions(luisa-compute-backend-cuda PRIVATE "NVTT_DEPRECATED_API=")
            # TODO: fix rpath on Linux?
            add_custom_command(TARGET luisa-compute-backend-cuda POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${NVTT_RUNTIME_LIB}" "$<TARGET_FILE_DIR:luisa-compute-core>")
            install(FILES "${NVTT_RUNTIME_LIB}" DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif ()
    endif ()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "CUDA not found. The CUDA backend will not be built.")
else ()
    message(WARNING "CUDA not found. The CUDA backend will not be built.")
endif ()
