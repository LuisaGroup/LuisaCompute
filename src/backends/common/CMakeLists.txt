# OIDN Denoiser Extension
if(WIN32)
    set(oidn_DOWNLOAD_URL "https://github.com/OpenImageDenoise/oidn/releases/download/v2.1.0/oidn-2.1.0.x64.windows.zip")
elseif(UNIX AND NOT APPLE)
    set(oidn_DOWNLOAD_URL "https://github.com/OpenImageDenoise/oidn/releases/download/v2.1.0/oidn-2.1.0.x86_64.linux.tar.gz")
else()
    set(oidn_DOWNLOAD_URL "")
    message(STATUS "OpenImageDenoise not available on Apple Silicon")
endif()

# TODO: Mike fix Apple Silicon pls
if(oidn_DOWNLOAD_URL)
    message(STATUS "Downloading OpenImageDenoise from ${oidn_DOWNLOAD_URL}")
  
    include(FetchContent)
    FetchContent_Declare(OpenImageDenoise URL ${oidn_DOWNLOAD_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(OpenImageDenoise)
    FetchContent_GetProperties(OpenImageDenoise)
    # message(STATUS "openimagedenoise_BINARY_DIR: ${openimagedenoise_BINARY_DIR}")
    message(STATUS "OpenImageDenoise downloaded to ${openimagedenoise_SOURCE_DIR}")
    set(OpenImageDenoise_DIR ${openimagedenoise_SOURCE_DIR}/lib/cmake/OpenImageDenoise-2.1.0)
    find_package(OpenImageDenoise REQUIRED)
    add_library(luisa-compute-oidn-ext STATIC oidn_denoiser.cpp)
    target_link_libraries(luisa-compute-oidn-ext PUBLIC luisa-compute-runtime OpenImageDenoise)
    install(TARGETS luisa-compute-oidn-ext EXPORT LuisaComputeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # copy the  ${openimagedenoise_BINARY_DIR}/bin/*.dll to the build directory
    file(GLOB OIDN_DLLS
        ${openimagedenoise_SOURCE_DIR}/bin/*.dll
        ${openimagedenoise_SOURCE_DIR}/bin/*.so
        ${openimagedenoise_SOURCE_DIR}/bin/*.dylib)
    message(STATUS "Copying ${OIDN_DLLS} to ${CMAKE_INSTALL_BINDIR}")
    add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OIDN_DLLS}
        ${CMAKE_INSTALL_BINDIR})
endif()

#
if(LUISA_COMPUTE_ENABLE_CPU OR
    LUISA_COMPUTE_ENABLE_CUDA OR
    LUISA_COMPUTE_ENABLE_REMOTE)
    find_package(Vulkan)

    if(UNIX AND NOT APPLE)
        find_package(X11)
        set(X11_DEPENDENCY_SATISFIED ${X11_FOUND} INTERNAL)
    else()
        set(X11_DEPENDENCY_SATISFIED TRUE INTERNAL)
    endif()

    if(Vulkan_FOUND AND X11_DEPENDENCY_SATISFIED)
        set(LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
            vulkan_instance.cpp vulkan_instance.h
            vulkan_swapchain.cpp vulkan_swapchain.h)

        if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
            enable_language(OBJC)
            enable_language(OBJCXX)
            list(APPEND LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
                moltenvk_surface.mm)
        endif()

        add_library(luisa-compute-vulkan-swapchain SHARED
            ${LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES})

        target_link_libraries(luisa-compute-vulkan-swapchain PUBLIC
            luisa-compute-runtime
            Vulkan::Vulkan
            ${X11_LIBRARIES})
        target_compile_definitions(luisa-compute-vulkan-swapchain
            PUBLIC LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN=1
            PRIVATE LC_BACKEND_EXPORT_DLL)

        if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_libraries(luisa-compute-vulkan-swapchain
                PUBLIC "-framework QuartzCore" "-framework AppKit")
        endif()

        set_target_properties(luisa-compute-vulkan-swapchain PROPERTIES
            OUTPUT_NAME lc-vulkan-swapchain)
        install(TARGETS luisa-compute-vulkan-swapchain EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    else()
        message(WARNING "Vulkan not found, vulkan swapchain will not be enabled.")
        add_library(luisa-compute-vulkan-swapchain INTERFACE)
    endif()
endif()

if(LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)
    file(GLOB LUISA_HLSL_BUILTIN_SOURCES CONFIGURE_DEPENDS "hlsl/builtin/*.c")
    add_library(luisa-compute-hlsl-builtin MODULE ${LUISA_HLSL_BUILTIN_SOURCES})
    target_compile_definitions(luisa-compute-hlsl-builtin PRIVATE LC_HLSL_DLL)
    set_target_properties(luisa-compute-hlsl-builtin PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME lc-hlsl-builtin)
    add_dependencies(luisa-compute-backends luisa-compute-hlsl-builtin)
    install(TARGETS luisa-compute-hlsl-builtin EXPORT LuisaComputeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
