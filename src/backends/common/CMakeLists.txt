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
                vulkan_swapchain.h vulkan_swapchain.cpp)
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
                    PUBLIC "-framework QuartzCore")
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
    set(LUISA_HLSL_BUILTIN_SOURCES
            hlsl/builtin/accel_header.c
            hlsl/builtin/accel_process.c
            hlsl/builtin/bc6_encode_block.c
            hlsl/builtin/bc6_header.c
            hlsl/builtin/bc6_trymode_g10cs.c
            hlsl/builtin/bc6_trymode_le10cs.c
            hlsl/builtin/bc7_encode_block.c
            hlsl/builtin/bc7_header.c
            hlsl/builtin/bc7_trymode_02cs.c
            hlsl/builtin/bc7_trymode_137cs.c
            hlsl/builtin/bc7_trymode_456cs.c
            hlsl/builtin/compute_quad.c
            hlsl/builtin/copy_sign.c
            hlsl/builtin/determinant.c
            hlsl/builtin/hlsl_header.c
            hlsl/builtin/indirect.c
            hlsl/builtin/inverse.c
            hlsl/builtin/raytracing_header.c
            hlsl/builtin/resource_size.c
            hlsl/builtin/bindless_common.c
            hlsl/builtin/tex2d_bindless.c
            hlsl/builtin/tex3d_bindless.c)

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
