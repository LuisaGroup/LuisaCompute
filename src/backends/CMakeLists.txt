add_library(luisa-compute-backends INTERFACE)
function(luisa_compute_add_backend name)
    cmake_parse_arguments(BACKEND "" "" "SOURCES" ${ARGN})
    add_library(luisa-compute-backend-${name} MODULE ${BACKEND_SOURCES})
    target_link_libraries(luisa-compute-backend-${name} PRIVATE
                          luisa-compute-runtime
                          luisa-compute-compile)
    add_dependencies(luisa-compute-backends luisa-compute-backend-${name})
    set_target_properties(luisa-compute-backend-${name} PROPERTIES
                          UNITY_BUILD ON
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/backends/"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/backends/"
                          DEBUG_POSTFIX "")
endfunction()

option(LUISA_COMPUTE_ENABLE_DX ON "Enable DirectX backend")
option(LUISA_COMPUTE_ENABLE_LLVM ON "Enable LLVM backend")
option(LUISA_COMPUTE_ENABLE_METAL ON "Enable Metal backend")

if (LUISA_COMPUTE_ENABLE_DX)
    add_subdirectory(dx)
endif ()

if (LUISA_COMPUTE_ENABLE_LLVM)
    add_subdirectory(llvm)
endif ()

if (LUISA_COMPUTE_ENABLE_METAL)
    add_subdirectory(metal)
endif ()
