add_subdirectory(common)

add_library(luisa-compute-backends INTERFACE)
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(lc-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(lc-backend-${name} PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            luisa-compute-gui
            luisa-compute-vstl
            luisa-compute-backend-common)
    if(LUISA_COMPUTE_ENABLE_DSL)
        target_link_libraries(lc-backend-${name} PRIVATE luisa-compute-dsl)
    endif()
    add_dependencies(luisa-compute-backends lc-backend-${name})
    set_target_properties(lc-backend-${name} PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            DEBUG_POSTFIX "")
    install(TARGETS lc-backend-${name}
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

if (LUISA_COMPUTE_ENABLE_DX)
    add_subdirectory(dx)
endif ()

if (LUISA_COMPUTE_ENABLE_METAL)
    add_subdirectory(metal)
endif ()

if (LUISA_COMPUTE_ENABLE_CUDA)
    add_subdirectory(cuda)
endif ()

if (LUISA_COMPUTE_ENABLE_CPU)
    add_subdirectory(cpu)
endif ()

install(TARGETS luisa-compute-backends
        EXPORT LuisaComputeTargets)
