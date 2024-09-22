# TODO
if (WIN32)
    message(STATUS "Build with DirectX backend")

    file(GLOB_RECURSE LC_BACKEND_DX_SOURCES CONFIGURE_DEPENDS "*.cpp" "../common/hlsl/*.cpp")

    # download support SDK files
    set(LUISA_COMPUTE_DOWNLOADED_SDK_FILE "${CMAKE_CURRENT_LIST_DIR}/../../../scripts/downloaded_sdks.cmake")
    message(STATUS "Checking for downloaded SDK configuration file: ${LUISA_COMPUTE_DOWNLOADED_SDK_FILE}")
    if (EXISTS "${LUISA_COMPUTE_DOWNLOADED_SDK_FILE}")
        message(STATUS "Reading downloaded SDK configuration from ${LUISA_COMPUTE_DOWNLOADED_SDK_FILE}")
        include("${LUISA_COMPUTE_DOWNLOADED_SDK_FILE}")
    endif ()
    option(LUISA_COMPUTE_DX_SDK_DIR "The directory of DirectX SDK" "")
    if (LUISA_COMPUTE_DX_SDK_DIR)
        cmake_path(IS_RELATIVE LUISA_COMPUTE_DX_SDK_DIR LUISA_COMPUTE_DX_SDK_DIR_is_relative)
        if (LUISA_COMPUTE_DX_SDK_DIR_is_relative)
            set(LUISA_COMPUTE_DX_SDK_DIR "${CMAKE_SOURCE_DIR}/${LUISA_COMPUTE_DX_SDK_DIR}")
        endif ()
        cmake_path(NORMAL_PATH LUISA_COMPUTE_DX_SDK_DIR OUTPUT_VARIABLE LUISA_COMPUTE_DX_SDK_DIR)
        message(STATUS "DirectX SDK directory: ${LUISA_COMPUTE_DX_SDK_DIR}")
    endif ()
    if (NOT LUISA_COMPUTE_DX_SDK_DIR OR NOT EXISTS "${LUISA_COMPUTE_DX_SDK_DIR}")
        set(LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL "https://github.com/LuisaGroup/SDKs/releases/download/sdk/dx_sdk_20240920.zip")
        set(LUISA_COMPUTE_DX_SDK_SHA1 "4c8390d674f375e6676ba15ce452db59df88da8f")
        message(STATUS "Downloading DirectX SDK from ${LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL}")
        include(FetchContent)
        FetchContent_Declare(luisa_compute_dx_sdk
                URL ${LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL}
                URL_HASH SHA1=${LUISA_COMPUTE_DX_SDK_SHA1})
        FetchContent_MakeAvailable(luisa_compute_dx_sdk)
        FetchContent_GetProperties(luisa_compute_dx_sdk)
        set(LUISA_COMPUTE_DX_SDK_DIR ${luisa_compute_dx_sdk_SOURCE_DIR})
    endif ()

    # check if the SDK files exist
    set(LUISA_COMPUTE_DX_SDK_FILES
            DirectML.dll
            dstorage.dll
            dstoragecore.dll
            dxcompiler.dll
            dxil.dll)
    foreach (file ${LUISA_COMPUTE_DX_SDK_FILES})
        if (NOT EXISTS "${LUISA_COMPUTE_DX_SDK_DIR}/${file}")
            message(FATAL_ERROR "The required file ${file} is missing in the DirectX SDK directory (${LUISA_COMPUTE_DX_SDK_DIR}). \
            You can either manually download the SDK from ${LUISA_COMPUTE_DX_SDK_DOWNLOAD_URL} and extract it to the specified directory, \
            or set LUISA_COMPUTE_DX_SDK_DIR to an empty string to let the build script download the SDK for you.")
        endif ()
    endforeach ()

    luisa_compute_add_backend(dx
            SOURCES ${LC_BACKEND_DX_SOURCES}
            SUPPORT_DIR ${LUISA_COMPUTE_DX_SDK_DIR})
    target_precompile_headers(luisa-compute-backend-dx PRIVATE pch.h)
    target_include_directories(luisa-compute-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(luisa-compute-backend-dx PRIVATE D3D12 d3dcompiler dxgi luisa-compute-vstl DirectML)
    target_compile_definitions(luisa-compute-backend-dx PRIVATE UNICODE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_definitions(luisa-compute-backend-dx PRIVATE
                _XM_NO_INTRINSICS_
                m128_f32=vector4_f32
                m128_u32=vector4_u32)
    endif ()

    find_package(CUDAToolkit 11.7 QUIET)
    if (CUDAToolkit_FOUND)
        target_link_libraries(luisa-compute-backend-dx PRIVATE CUDA::cuda_driver)
        target_compile_definitions(luisa-compute-backend-dx PRIVATE LCDX_ENABLE_CUDA)
    endif ()

    if (TARGET luisa-compute-oidn-ext)
        target_link_libraries(luisa-compute-backend-dx PRIVATE luisa-compute-oidn-ext)
    endif ()

    find_package(ZLIB)
    if (ZLIB_FOUND)
        target_link_libraries(luisa-compute-backend-dx PRIVATE ZLIB::ZLIB)
    else () # use maxwell's favorite xrepo otherwise
        target_include_directories(luisa-compute-backend-dx PRIVATE "../../../xrepo/packages/z/zlib/port/zlib/include")
        # convert zlib path to absolute path
        set(ZLIB_LIB_FILE "../../../xrepo/packages/z/zlib/port/zlib/lib/windows/x64/zlibstatic.lib")
        cmake_path(ABSOLUTE_PATH ZLIB_LIB_FILE)
        target_link_libraries(luisa-compute-backend-dx PRIVATE ${ZLIB_LIB_FILE})
    endif ()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "The DirectX backend will not be built on non-Windows platforms.")
else ()
    message(WARNING "The DirectX backend will not be built on non-Windows platforms.")
endif ()
