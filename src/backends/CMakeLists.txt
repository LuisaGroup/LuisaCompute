add_library(luisa-compute-backends INTERFACE)
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
            luisa-compute-ast
            luisa-compute-runtime
            luisa-compute-compile
            luisa-compute-vstl)
    add_dependencies(luisa-compute-backends luisa-compute-backend-${name})
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/backends/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/backends/"
            UNITY_BUILD ON
            DEBUG_POSTFIX "")
    foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set_target_properties(luisa-compute-backend-${name} PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/backends/"
                RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/backends/"
                UNITY_BUILD ON)
    endforeach ()
    string(TOUPPER ${name} UPPER_NAME)
    target_compile_definitions(luisa-compute-backends INTERFACE LUISA_BACKEND_${UPPER_NAME}_ENABLED)
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

if (LUISA_COMPUTE_ENABLE_ISPC)
    add_subdirectory(ispc)
endif ()

if (LUISA_COMPUTE_ENABLE_REMOTE)
    add_subdirectory(remote)
endif ()
