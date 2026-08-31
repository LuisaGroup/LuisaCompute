add_library(luisa-compute-backends INTERFACE)

option(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV "Enable XIR-to-SPIRV codegen path for the Vulkan backend" ON)
if (LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV AND LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV)
    message(FATAL_ERROR "Vulkan compute shader codegen options are mutually exclusive. "
            "Use LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV for the default native SPIR-V path, "
            "or LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV for the experimental LLVM SPIR-V path.")
endif ()

set(FETCHCONTENT_QUIET OFF)
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "SUPPORT_DIR;BUILTIN_DIR" "SOURCES" ${ARGN})
    # DLL target
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            luisa-compute-gui)
    if (LUISA_COMPUTE_ENABLE_DSL)
        target_link_libraries(luisa-compute-backend-${name} PRIVATE luisa-compute-dsl)
    endif ()
    add_dependencies(luisa-compute-backends luisa-compute-backend-${name})
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            DEBUG_POSTFIX ""
            EXPORT_NAME backend_${name}
            OUTPUT_NAME luisa-backend-${name})
    set_property(GLOBAL APPEND PROPERTY LUISA_COMPUTE_BUILT_BACKENDS "${name}")
    install(TARGETS luisa-compute-backend-${name}
            EXPORT LuisaComputeTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    # support directory
    if (BACKEND_SUPPORT_DIR)
        add_custom_target(luisa-compute-backend-${name}-copy-support ALL
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${BACKEND_SUPPORT_DIR}"
                "$<TARGET_FILE_DIR:luisa-compute-core>/")
        add_dependencies(luisa-compute-backend-${name} luisa-compute-backend-${name}-copy-support)
        install(DIRECTORY ${BACKEND_SUPPORT_DIR}/
                DESTINATION ${CMAKE_INSTALL_BINDIR}/)
    endif ()
endfunction()

add_subdirectory(common)
add_subdirectory(validation)

if (LUISA_COMPUTE_ENABLE_DX)
    add_subdirectory(dx)
endif ()

if (LUISA_COMPUTE_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

if (LUISA_COMPUTE_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

if (LUISA_COMPUTE_ENABLE_HIP)
    add_subdirectory(hip)
endif ()

if (LUISA_COMPUTE_ENABLE_VULKAN)
    add_subdirectory(vk)
endif ()

if (LUISA_COMPUTE_ENABLE_FALLBACK)
    add_subdirectory(fallback)
endif ()

if (LUISA_COMPUTE_ENABLE_SIMD)
    add_subdirectory(simd)
endif ()

install(TARGETS luisa-compute-backends
        EXPORT LuisaComputeTargets)
