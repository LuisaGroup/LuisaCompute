# OIDN Denoiser Extension
if (LUISA_COMPUTE_DOWNLOAD_OIDN)

    set(oidn_DOWNLOAD_VERSION 2.2.2)
    if (WIN32)
        set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x64.windows.zip")
    elseif (UNIX AND NOT APPLE)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.aarch64.linux.tar.gz")
        else()
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x86_64.linux.tar.gz")
        endif()
    elseif (APPLE)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.x86_64.macos.tar.gz")
        else ()
            set(oidn_DOWNLOAD_URL "https://github.com/RenderKit/oidn/releases/download/v${oidn_DOWNLOAD_VERSION}/oidn-${oidn_DOWNLOAD_VERSION}.arm64.macos.tar.gz")
        endif ()
    endif ()

    message(STATUS "Downloading OpenImageDenoise from ${oidn_DOWNLOAD_URL}")

    include(FetchContent)
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

    add_library(luisa-compute-oidn-ext STATIC oidn_denoiser.cpp)
    set_target_properties(luisa-compute-oidn-ext PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # fix the rpaths of the OpenImageDenoise libraries
    if (APPLE) # recent version already has the correct RPATH
        # find_program(INSTALL_NAME_TOOL_EXE install_name_tool)
        # if (INSTALL_NAME_TOOL_EXE)
        #     foreach (oidn_lib ${oidn_RUNTIME_LIBS})
        #         # get file name without path
        #         get_filename_component(oidn_lib_name ${oidn_lib} NAME)
        #         add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
        #                 COMMENT "Fixing rpath of ${oidn_lib_name}"
        #                 COMMAND ${INSTALL_NAME_TOOL_EXE} -id "@rpath/${oidn_lib_name}" "${oidn_lib}"
        #                 COMMAND ${INSTALL_NAME_TOOL_EXE} -add_rpath "@loader_path" "${oidn_lib}"
        #                 BYPRODUCTS ${oidn_lib}
        #                 VERBATIM)
        #     endforeach ()
        # else ()
        #     message(FATAL_ERROR "install_name_tool not found, rpath of OpenImageDenoise libraries will not be fixed. You may either install Xcode or disable the OIDN extension with -DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF when configuring.")
        # endif ()
    elseif (UNIX)
        find_program(PATCHELF_EXE patchelf)
        if (PATCHELF_EXE)
            foreach (oidn_lib ${oidn_RUNTIME_LIBS})
                add_custom_command(TARGET luisa-compute-oidn-ext POST_BUILD
                        COMMENT "Fixing rpath of ${oidn_lib}"
                        COMMAND ${PATCHELF_EXE} --set-rpath "$ORIGIN" "${oidn_lib}"
                        BYPRODUCTS ${oidn_lib}
                        VERBATIM)
            endforeach ()
        else ()
            message(FATAL_ERROR "patchelf not found, rpath of OpenImageDenoise libraries will not be fixed. You may either install patchelf or disable the OIDN extension with -DLUISA_COMPUTE_DOWNLOAD_OIDN=OFF when configuring.")
        endif ()
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
        LUISA_COMPUTE_ENABLE_REMOTE OR
        LUISA_COMPUTE_ENABLE_FALLBACK)

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
                vulkan_swapchain.cpp)

        if (APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
            enable_language(OBJC)
            enable_language(OBJCXX)
            list(APPEND LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
                    moltenvk_surface.mm)
        endif ()

        add_library(luisa-compute-vulkan-swapchain OBJECT
                ${LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES})
        target_link_libraries(luisa-compute-vulkan-swapchain
                PUBLIC
                luisa-compute-runtime
                Vulkan::Vulkan
                ${X11_LIBRARIES})
        target_compile_definitions(luisa-compute-vulkan-swapchain
                PUBLIC
                LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN=1)

        if (UNIX AND NOT APPLE)
            set_target_properties(luisa-compute-vulkan-swapchain PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif ()

        if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_link_libraries(luisa-compute-vulkan-swapchain
                    PUBLIC "-framework QuartzCore" "-framework AppKit")
        endif ()

        # wayland support
        if (UNIX AND NOT APPLE AND LUISA_COMPUTE_ENABLE_WAYLAND)
            find_package(PkgConfig REQUIRED)
            pkg_check_modules(Wayland REQUIRED wayland-client)
            target_include_directories(luisa-compute-vulkan-swapchain PRIVATE ${Wayland_INCLUDE_DIRS})
            target_link_libraries(luisa-compute-vulkan-swapchain PRIVATE ${Wayland_LIBRARIES})
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
