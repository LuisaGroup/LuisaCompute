add_library(luisa-compute-backends INTERFACE)
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
            OUTPUT_NAME lc-backend-${name})
    install(TARGETS luisa-compute-backend-${name}
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

if (LUISA_COMPUTE_ENABLE_RUST)
    if (LUISA_COMPUTE_ENABLE_CPU)
        add_subdirectory(cpu)
    endif ()

    if (LUISA_COMPUTE_ENABLE_REMOTE)
        add_subdirectory(remote)
    endif ()
endif ()

install(TARGETS luisa-compute-backends
        EXPORT LuisaComputeTargets)
