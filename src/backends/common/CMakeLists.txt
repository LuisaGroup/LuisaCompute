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
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
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
    install(TARGETS luisa-compute-hlsl-builtin EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()
