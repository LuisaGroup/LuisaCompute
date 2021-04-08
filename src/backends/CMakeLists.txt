add_library(luisa-compute-backends INTERFACE)
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
                          luisa-compute-ast
                          luisa-compute-runtime
                          luisa-compute-compile)
    add_dependencies(luisa-compute-backends luisa-compute-backend-${name})
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/backends"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/backends"
                          UNITY_BUILD ON
                          DEBUG_POSTFIX "")
    foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG)
        set_target_properties(luisa-compute-backend-${name} PROPERTIES
                              LIBRARY_OUTPUT_DIRECTORY_${CONFIG} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG}}/backends"
                              RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG}}/backends")
    endforeach ()
    string(TOUPPER ${name} UPPER_NAME)
    target_compile_definitions(luisa-compute-backends INTERFACE LUISA_BACKEND_${UPPER_NAME}_ENABLED)
endfunction()

option(LUISA_COMPUTE_ENABLE_DX "Enable DirectX backend" ON)
option(LUISA_COMPUTE_ENABLE_LLVM "Enable LLVM backend" ON)
option(LUISA_COMPUTE_ENABLE_METAL "Enable Metal backend" ON)

if (LUISA_COMPUTE_ENABLE_DX)
    add_subdirectory(dx)
endif ()

if (LUISA_COMPUTE_ENABLE_LLVM)
    add_subdirectory(llvm)
endif ()

if (LUISA_COMPUTE_ENABLE_METAL)
    add_subdirectory(metal)
endif ()
