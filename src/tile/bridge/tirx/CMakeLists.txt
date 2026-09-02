find_path(LUISA_TILE_TVM_INCLUDE_PATH
        NAMES tvm/tirx/function.h
        HINTS "${LUISA_COMPUTE_TVM_INCLUDE_DIR}"
        NO_DEFAULT_PATH)
find_path(LUISA_TILE_TVM_FFI_INCLUDE_PATH
        NAMES tvm/ffi/object.h
        HINTS "${LUISA_COMPUTE_TVM_FFI_INCLUDE_DIR}"
        NO_DEFAULT_PATH)
find_library(LUISA_TILE_TVM_COMPILER_LIBRARY
        NAMES tvm_compiler
        HINTS "${LUISA_COMPUTE_TVM_LIBRARY_DIR}"
        NO_DEFAULT_PATH)
find_library(LUISA_TILE_TVM_RUNTIME_LIBRARY
        NAMES tvm_runtime
        HINTS "${LUISA_COMPUTE_TVM_LIBRARY_DIR}"
        NO_DEFAULT_PATH)
find_library(LUISA_TILE_TVM_FFI_LIBRARY
        NAMES tvm_ffi
        HINTS "${LUISA_COMPUTE_TVM_FFI_LIBRARY_DIR}"
        NO_DEFAULT_PATH)

foreach (_dependency
        LUISA_TILE_TVM_INCLUDE_PATH
        LUISA_TILE_TVM_FFI_INCLUDE_PATH
        LUISA_TILE_TVM_COMPILER_LIBRARY
        LUISA_TILE_TVM_RUNTIME_LIBRARY
        LUISA_TILE_TVM_FFI_LIBRARY)
    if (NOT DEFINED ${_dependency} OR
        "${${_dependency}}" STREQUAL "" OR
        "${${_dependency}}" MATCHES "-NOTFOUND$")
        message(FATAL_ERROR
                "LUISA_COMPUTE_ENABLE_TILE_TIRX_BRIDGE requires ${_dependency}. "
                "Set the LUISA_COMPUTE_TVM_* paths to a TVM development installation.")
    endif ()
endforeach ()

set(LUISA_COMPUTE_TILE_TIRX_BRIDGE_SOURCES
        compiler.cpp
        layout.cpp
        lower.cpp)

add_library(luisa-compute-tile-bridge-tirx
        ${LUISA_COMPUTE_TILE_TIRX_BRIDGE_SOURCES})
target_compile_definitions(luisa-compute-tile-bridge-tirx
        PRIVATE LUISA_TILE_TIRX_BRIDGE_EXPORT_DLL=1)
target_include_directories(luisa-compute-tile-bridge-tirx
        PUBLIC
        $<BUILD_INTERFACE:${LUISA_TILE_TVM_INCLUDE_PATH}>
        $<BUILD_INTERFACE:${LUISA_TILE_TVM_FFI_INCLUDE_PATH}>)
target_link_libraries(luisa-compute-tile-bridge-tirx
        PUBLIC
        luisa-compute-tile
        "${LUISA_TILE_TVM_COMPILER_LIBRARY}"
        "${LUISA_TILE_TVM_RUNTIME_LIBRARY}"
        "${LUISA_TILE_TVM_FFI_LIBRARY}")
if (UNIX)
    target_link_options(luisa-compute-tile-bridge-tirx PUBLIC
            "LINKER:-rpath,${LUISA_COMPUTE_TVM_LIBRARY_DIR}"
            "LINKER:-rpath,${LUISA_COMPUTE_TVM_FFI_LIBRARY_DIR}")
endif ()
set_target_properties(luisa-compute-tile-bridge-tirx PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
        OUTPUT_NAME luisa-tile-bridge-tirx)

add_library(luisa::tile-tirx ALIAS luisa-compute-tile-bridge-tirx)
