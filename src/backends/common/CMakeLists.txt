# OIDN Denoiser Extension
if (LUISA_COMPUTE_DOWNLOAD_OIDN)

    set(oidn_DOWNLOAD_VERSION 2.1.0)
    set(oidn_BUILD_FROM_SOURCE OFF)
    if (WIN32)
        set(oidn_DOWNLOAD_URL "https://github.com/OpenImageDenoise/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x64.windows.zip")
    elseif (UNIX AND NOT APPLE)
        set(oidn_DOWNLOAD_URL "https://github.com/OpenImageDenoise/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x86_64.linux.tar.gz")
    elseif (APPLE)
        set(oidn_BUILD_FROM_SOURCE ON)
        set(oidn_DOWNLOAD_URL "https://github.com/OpenImageDenoise/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.src.tar.gz")
    endif ()

    message(STATUS "Downloading OpenImageDenoise from ${oidn_DOWNLOAD_URL}")

    include(FetchContent)
    if (oidn_BUILD_FROM_SOURCE) # FIXME: currently this is only for macOS
        FetchContent_Declare(OpenImageDenoise
                URL ${oidn_DOWNLOAD_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP ON
                OVERRIDE_FIND_PACKAGE)
        set(OIDN_INSTALL_DEPENDENCIES ON CACHE BOOL "" FORCE)
        set(OIDN_APPS OFF CACHE BOOL "" FORCE)
        find_package(OpenImageDenoise CONFIG REQUIRED)
        FetchContent_GetProperties(OpenImageDenoise)
        message(STATUS "OpenImageDenoise source downloaded to ${openimagedenoise_SOURCE_DIR}")
        string(REPLACE "." ";" oidn_DOWNLOAD_VERSION_LIST ${oidn_DOWNLOAD_VERSION})
        list(GET oidn_DOWNLOAD_VERSION_LIST 0 oidn_DOWNLOAD_VERSION_MAJOR)
        # FIXME: anyway to glob these runtime libs?
        set(oidn_RUNTIME_LIBS
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise.${oidn_DOWNLOAD_VERSION}.dylib"
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise.${oidn_DOWNLOAD_VERSION_MAJOR}.dylib"
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise.dylib"
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise_core.${oidn_DOWNLOAD_VERSION}.dylib"
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise_core.dylib"
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise_device_cpu.${oidn_DOWNLOAD_VERSION}.dylib"
                "${openimagedenoise_BINARY_DIR}/libOpenImageDenoise_device_cpu.dylib")
    else ()
        FetchContent_Declare(OpenImageDenoise
                URL ${oidn_DOWNLOAD_URL}
                DOWNLOAD_EXTRACT_TIMESTAMP ON)
        FetchContent_MakeAvailable(OpenImageDenoise)
        FetchContent_GetProperties(OpenImageDenoise)
        message(STATUS "OpenImageDenoise binary downloaded to ${openimagedenoise_SOURCE_DIR}")
        list(APPEND CMAKE_PREFIX_PATH ${openimagedenoise_SOURCE_DIR})
        find_package(OpenImageDenoise CONFIG REQUIRED)
        file(GLOB oidn_RUNTIME_LIBS
                "${openimagedenoise_SOURCE_DIR}/bin/*.dll"
                "${openimagedenoise_SOURCE_DIR}/lib/*.so"
                "${openimagedenoise_SOURCE_DIR}/lib/*.so.*"
                "${openimagedenoise_SOURCE_DIR}/lib/*.dylib")
        # set(oidn_RUNTIME_BACKEND_LIBS ${oidn_RUNTIME_LIBS})
        # list(FILTER oidn_RUNTIME_BACKEND_LIBS INCLUDE REGEX ".*_device_(cpu|cuda).*")
        # list(FILTER oidn_RUNTIME_LIBS EXCLUDE REGEX ".*_device_.*")
        # list(APPEND oidn_RUNTIME_LIBS ${oidn_RUNTIME_BACKEND_LIBS})
    endif ()

    if (oidn_BUILD_FROM_SOURCE)
        # FIXME: trouble linking the STATIC library on macOS with Apple Silicons, so SHARED here to workaround...
        add_library(luisa-compute-oidn-ext SHARED oidn_denoiser.cpp)
        set_target_properties(luisa-compute-oidn-ext PROPERTIES OUTPUT_NAME lc-oidn-ext)
        # FIXME: ad-hoc fix for the missing dependency of the OpenImageDenoise_device_cpu library
        add_dependencies(luisa-compute-oidn-ext OpenImageDenoise_device_cpu)
    else ()
        add_library(luisa-compute-oidn-ext STATIC oidn_denoiser.cpp)
        set_target_properties(luisa-compute-oidn-ext PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif ()
    target_compile_definitions(luisa-compute-oidn-ext
            PUBLIC LUISA_BACKEND_ENABLE_OIDN=1
            PRIVATE LC_BACKEND_EXPORT_DLL)
    target_link_libraries(luisa-compute-oidn-ext PUBLIC
            luisa-compute-runtime
            OpenImageDenoise)

    # copy the OpenImageDenoise dll to the output directory
    message(STATUS "OpenImageDenoise runtime libs: ${oidn_RUNTIME_LIBS}")
    foreach (oidn_RUNTIME_LIB ${oidn_RUNTIME_LIBS})
        add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${oidn_RUNTIME_LIB}"
                "$<TARGET_FILE_DIR:luisa-compute-core>")
    endforeach ()
    install(TARGETS luisa-compute-oidn-ext EXPORT LuisaComputeTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES ${oidn_RUNTIME_LIBS} DESTINATION ${CMAKE_INSTALL_BINDIR})

endif ()

if (LUISA_COMPUTE_ENABLE_CPU OR
        LUISA_COMPUTE_ENABLE_CUDA OR
        LUISA_COMPUTE_ENABLE_REMOTE)

    find_package(Vulkan)

    if (UNIX AND NOT APPLE)
        find_package(X11)
        set(X11_DEPENDENCY_SATISFIED ${X11_FOUND} INTERNAL)
    else ()
        set(X11_DEPENDENCY_SATISFIED TRUE INTERNAL)
    endif ()

    if (Vulkan_FOUND AND X11_DEPENDENCY_SATISFIED)
        set(LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
                vulkan_instance.cpp vulkan_instance.h
                vulkan_swapchain.cpp vulkan_swapchain.h)

        if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
            enable_language(OBJC)
            enable_language(OBJCXX)
            list(APPEND LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
                    moltenvk_surface.mm)
        endif ()

        add_library(luisa-compute-vulkan-swapchain SHARED
                ${LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES})

        target_link_libraries(luisa-compute-vulkan-swapchain PUBLIC
                luisa-compute-runtime
                Vulkan::Vulkan
                ${X11_LIBRARIES})
        target_compile_definitions(luisa-compute-vulkan-swapchain
                PUBLIC LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN=1
                PRIVATE LC_BACKEND_EXPORT_DLL)

        if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_libraries(luisa-compute-vulkan-swapchain
                    PUBLIC "-framework QuartzCore" "-framework AppKit")
        endif ()

        set_target_properties(luisa-compute-vulkan-swapchain PROPERTIES
                OUTPUT_NAME lc-vulkan-swapchain)
        install(TARGETS luisa-compute-vulkan-swapchain EXPORT LuisaComputeTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    else ()
        message(WARNING "Vulkan not found, vulkan swapchain will not be enabled.")
        add_library(luisa-compute-vulkan-swapchain INTERFACE)
    endif ()
endif ()

if (LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)
    file(GLOB LUISA_HLSL_BUILTIN_SOURCES CONFIGURE_DEPENDS "hlsl/builtin/*.c")
    add_library(luisa-compute-hlsl-builtin MODULE ${LUISA_HLSL_BUILTIN_SOURCES})
    target_compile_definitions(luisa-compute-hlsl-builtin PRIVATE LC_HLSL_DLL)
    set_target_properties(luisa-compute-hlsl-builtin PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-hlsl-builtin)
    add_dependencies(luisa-compute-backends luisa-compute-hlsl-builtin)
    install(TARGETS luisa-compute-hlsl-builtin
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()
