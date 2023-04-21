if (LUISA_COMPUTE_ENABLE_CPU OR LUISA_COMPUTE_ENABLE_REMOTE OR LUISA_COMPUTE_COMPILED_BY_RUST_FRONTEND)
    find_package(Vulkan)
    find_package(X11)
    set(LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES
            vulkan_swapchain.h vulkan_swapchain.cpp
            )
    add_library(luisa-compute-vulkan-swapchain SHARED ${LUISA_COMPUTE_VULKAN_SWAPCHAIN_SOURCES})
    target_link_libraries(luisa-compute-vulkan-swapchain PRIVATE luisa-compute-runtime)
    target_link_libraries(luisa-compute-vulkan-swapchain PRIVATE Vulkan::Vulkan ${X11_LIBRARIES})

    if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        enable_language(OBJC)
        enable_language(OBJCXX)
        target_sources(luisa-compute-vulkan-swapchain PRIVATE moltenvk_surface.mm)
        target_link_libraries(luisa-compute-vulkan-swapchain PRIVATE "-framework QuartzCore")
    endif ()
    set_target_properties(luisa-compute-vulkan-swapchain PROPERTIES
            OUTPUT_NAME lc-vulkan-swapchain)
endif ()

if (LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)
    set(LUISA_HLSL_BUILTIN_SOURCES
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
            hlsl/builtin/hlsl_builtin.h
            hlsl/builtin/hlsl_config.h
            hlsl/builtin/hlsl_header.c
            hlsl/builtin/raytracing_header.c)
    add_library(luisa-compute-hlsl-builtin MODULE ${LUISA_HLSL_BUILTIN_SOURCES})
    target_compile_definitions(luisa-compute-hlsl-builtin PRIVATE LC_HLSL_DLL)
    set_target_properties(luisa-compute-hlsl-builtin PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-hlsl-builtin)
    add_dependencies(luisa-compute-backends luisa-compute-hlsl-builtin)
endif ()
